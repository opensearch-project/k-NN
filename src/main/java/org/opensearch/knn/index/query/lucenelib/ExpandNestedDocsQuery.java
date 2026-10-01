/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Builder;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This query is for a nested k-NN field to return multiple nested field documents
 * rather than only the highest-scoring nested field document.
 *
 * It begins by performing an approximate nearest neighbor search. Once results are gathered from all segments,
 * they are reduced to the top k results. Then, it constructs filtered document IDs for nested field documents
 * from these top k parent documents. Using these document IDs, it executes an exact nearest neighbor search
 * with a k value of Integer.MAX_VALUE, which provides scores for all specified nested field documents.
 *
 * When rescoring is enabled, two extra stages run between the approximate search and the expansion. The
 * approximate search walked the graph over quantized vectors, so every parent's representative child was
 * elected on inexact scores. The rescore stage re-scores all siblings of the oversampled candidates against
 * full precision vectors, collapses each parent group back to its single best child, and cuts the result to
 * k. Expanding only after that cut is what keeps the list at one row per document until the final top-k is
 * decided; cutting to k after the expansion would count child documents instead of parent documents and drop
 * parents that belong in the result. This mirrors the ordering that
 * {@link org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery} uses for the native engines.
 */
@Builder
public class ExpandNestedDocsQuery extends Query {
    final private InternalNestedKnnVectorQuery internalNestedKnnVectorQuery;
    final private QueryUtils queryUtils;
    /**
     * Number of oversampled parent candidates the approximate search kept for rescoring.
     * {@link RescoreContext#NO_RESCORE_NEEDED} means no rescoring, in which case the expansion runs
     * directly on the approximate search results.
     */
    @Builder.Default
    final private int rescoreK = RescoreContext.NO_RESCORE_NEEDED;
    /**
     * Full precision query vector. Required when {@link #rescoreK} enables rescoring, otherwise unused.
     */
    final private float[] floatQueryVector;
    /**
     * Runs the full precision rescore pass. Optional test seam; when unset one is created on demand, and only
     * when {@link #rescoreK} enables rescoring. It is not created up front because building it reaches for the
     * {@link ModelDao} singleton, which is only usable once the plugin has been initialized on a node.
     */
    final private ExactSearcher exactSearcher;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Query docAndScoreQuery = internalNestedKnnVectorQuery.knnRewrite(searcher);
        Weight weight = docAndScoreQuery.createWeight(searcher, scoreMode, boost);
        IndexReader reader = searcher.getIndexReader();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        Weight filterWeight = getFilterWeight(searcher);
        // Both the rescore and the expansion stage need the filter bits of every leaf they touch, so they are
        // built once per leaf and shared instead of being rebuilt by each stage.
        Map<Integer, Bits> filterBitsByLeaf = new ConcurrentHashMap<>();
        ContextualProfileBreakdown profile = getProfileBreakdown(searcher);
        List<Map<Integer, Float>> perLeafResults;
        perLeafResults = queryUtils.doSearch(searcher, leafReaderContexts, weight);
        if (isRescoreEnabled()) {
            perLeafResults = rescoreToTopKParents(searcher, leafReaderContexts, perLeafResults, filterWeight, filterBitsByLeaf, profile);
        }
        TopDocs[] topDocs = retrieveAll(searcher, leafReaderContexts, perLeafResults, filterWeight, filterBitsByLeaf, profile);
        int sum = 0;
        for (TopDocs topDoc : topDocs) {
            sum += topDoc.scoreDocs.length;
        }
        TopDocs topK = TopDocs.merge(sum, topDocs);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }
        return queryUtils.createDocAndScoreQuery(reader, topK).createWeight(searcher, scoreMode, boost);
    }

    private boolean isRescoreEnabled() {
        return rescoreK != RescoreContext.NO_RESCORE_NEEDED;
    }

    private ExactSearcher resolveExactSearcher() {
        return exactSearcher != null ? exactSearcher : new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance());
    }

    /**
     * Returns the profile breakdown for this query, or null when the search is not being profiled. The node for
     * this query is added to the profile tree by
     * {@link org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery#createWeight} before it delegates
     * here, so the lookup always resolves. The breakdown only carries the metrics that
     * {@link org.opensearch.knn.plugin.KNNPlugin#getQueryProfileMetricsProvider} registers for this query type.
     */
    private ContextualProfileBreakdown getProfileBreakdown(final IndexSearcher indexSearcher) {
        QueryProfiler profiler = KNNProfileUtil.getProfiler(indexSearcher);
        if (profiler == null) {
            return null;
        }
        return (ContextualProfileBreakdown) profiler.getProfileBreakdown(this);
    }

    /**
     * Re-scores the oversampled candidates against full precision vectors and reduces them to the top k
     * parent documents.
     *
     */
    private List<Map<Integer, Float>> rescoreToTopKParents(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final List<Map<Integer, Float>> perLeafResults,
        final Weight filterWeight,
        final Map<Integer, Bits> filterBitsByLeaf,
        final ContextualProfileBreakdown profile
    ) throws IOException {
        final int k = internalNestedKnnVectorQuery.getK();
        final ExactSearcher searcher = resolveExactSearcher();
        List<Callable<TopDocs>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            Map<Integer, Float> leafResult = perLeafResults.get(i);
            int leafOrd = i;
            rescoreTasks.add(() -> {
                if (leafResult.isEmpty()) {
                    return NestedKnnUtil.EMPTY_TOP_DOCS;
                }
                Bits queryFilter = filterBits(filterBitsByLeaf, leafReaderContext, filterWeight);
                DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
                    leafReaderContext,
                    leafResult.keySet(),
                    internalNestedKnnVectorQuery.getParentFilter(),
                    queryFilter
                );
                TopDocs leafTopDocs = QueryUtils.rescoreLeafWithFullPrecision(
                    searcher,
                    profile,
                    leafReaderContext,
                    internalNestedKnnVectorQuery.getField(),
                    floatQueryVector,
                    allSiblings,
                    k,
                    // passing the parent filter makes the searcher collapse each parent group to its best child
                    internalNestedKnnVectorQuery.getParentFilter()
                );
                for (ScoreDoc scoreDoc : leafTopDocs.scoreDocs) {
                    scoreDoc.shardIndex = leafOrd;
                }
                return leafTopDocs;
            });
        }
        TopDocs[] perLeafTopDocs = indexSearcher.getTaskExecutor().invokeAll(rescoreTasks).toArray(TopDocs[]::new);
        TopDocs topKParents = TopDocs.merge(k, perLeafTopDocs);

        List<Map<Integer, Float>> reducedResults = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            reducedResults.add(new HashMap<>());
        }
        for (ScoreDoc scoreDoc : topKParents.scoreDocs) {
            reducedResults.get(scoreDoc.shardIndex).put(scoreDoc.doc, scoreDoc.score);
        }
        return reducedResults;
    }

    private TopDocs[] retrieveAll(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final List<Map<Integer, Float>> perLeafResults,
        final Weight filterWeight,
        final Map<Integer, Bits> filterBitsByLeaf,
        final ContextualProfileBreakdown profile
    ) throws IOException {
        // Construct query
        List<Callable<TopDocs>> nestedQueryTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            nestedQueryTasks.add(() -> {
                Bits queryFilter = filterBits(filterBitsByLeaf, leafReaderContext, filterWeight);
                DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
                    leafReaderContext,
                    perLeafResults.get(finalI).keySet(),
                    internalNestedKnnVectorQuery.getParentFilter(),
                    queryFilter
                );
                TopDocs topDocs = scoreAllSiblings(leafReaderContext, allSiblings, profile);
                // Update doc id from segment id to shard id
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    scoreDoc.doc = scoreDoc.doc + leafReaderContext.docBase;
                }
                return topDocs;
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(nestedQueryTasks).toArray(TopDocs[]::new);
    }

    /**
     * Scores every sibling of the surviving parents, each keeping its own score, with no collapsing.
     *
     */
    private TopDocs scoreAllSiblings(
        final LeafReaderContext leafReaderContext,
        final DocIdSetIterator allSiblings,
        final ContextualProfileBreakdown profile
    ) throws IOException {
        if (isRescoreEnabled() == false) {
            return (TopDocs) KNNProfileUtil.profileBreakdown(
                profile,
                leafReaderContext,
                KNNQueryTimingType.EXACT_SEARCH,
                () -> internalNestedKnnVectorQuery.knnExactSearch(leafReaderContext, allSiblings)
            );
        }
        return QueryUtils.rescoreLeafWithFullPrecision(
            resolveExactSearcher(),
            profile,
            leafReaderContext,
            internalNestedKnnVectorQuery.getField(),
            floatQueryVector,
            allSiblings,
            // every sibling keeps its own score, so no top-k cut and no parent filter here
            (int) allSiblings.cost(),
            null
        );
    }

    /**
     * Returns the filter bits of a leaf, building them on first use. Reused across the rescore and expansion
     * stages, which would otherwise each rebuild the same bit set.
     */
    private Bits filterBits(final Map<Integer, Bits> filterBitsByLeaf, final LeafReaderContext leafReaderContext, final Weight filterWeight)
        throws IOException {
        Bits cached = filterBitsByLeaf.get(leafReaderContext.ord);
        if (cached != null) {
            return cached;
        }
        Bits bits = queryUtils.createBits(leafReaderContext, filterWeight);
        if (bits == null) {
            // Nothing worth caching, and the map does not accept null values. Callers treat null as "no filter".
            return null;
        }
        Bits existing = filterBitsByLeaf.putIfAbsent(leafReaderContext.ord, bits);
        return existing != null ? existing : bits;
    }

    /**
     * This is copied from {@link org.apache.lucene.search.AbstractKnnVectorQuery#rewrite}
     */
    private Weight getFilterWeight(final IndexSearcher indexSearcher) throws IOException {
        if (internalNestedKnnVectorQuery.getFilter() == null) {
            return null;
        }

        BooleanQuery booleanQuery = (new BooleanQuery.Builder()).add(internalNestedKnnVectorQuery.getFilter(), BooleanClause.Occur.FILTER)
            .add(new FieldExistsQuery(internalNestedKnnVectorQuery.getField()), BooleanClause.Occur.FILTER)
            .build();
        Query rewritten = indexSearcher.rewrite(booleanQuery);
        return indexSearcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1.0F);
    }

    @Override
    public void visit(final QueryVisitor queryVisitor) {
        queryVisitor.visitLeaf(this);
    }

    @Override
    public boolean equals(final Object o) {
        if (!sameClassAs(o)) {
            return false;
        }
        ExpandNestedDocsQuery other = (ExpandNestedDocsQuery) o;
        // rescoreK is not part of internalNestedKnnVectorQuery's equality, so it has to be compared here.
        // Otherwise two queries differing only in oversample_factor would be considered equal by the query cache.
        return internalNestedKnnVectorQuery.equals(other.internalNestedKnnVectorQuery) && rescoreK == other.rescoreK;
    }

    @Override
    public int hashCode() {
        return Objects.hash(internalNestedKnnVectorQuery, rescoreK);
    }

    @Override
    public String toString(final String s) {
        return this.getClass().getSimpleName()
            + "["
            + internalNestedKnnVectorQuery.getField()
            + "]..."
            + internalNestedKnnVectorQuery.getClass().getSimpleName()
            + "["
            + internalNestedKnnVectorQuery.toString()
            + "]";
    }
}
