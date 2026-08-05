/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Builder;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
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
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * This query is for a nested k-NN field to return multiple nested field documents
 * rather than only the highest-scoring nested field document.
 *
 * It begins by performing an approximate nearest neighbor search. Once results are gathered from all segments,
 * they are reduced to the top k results. Then, it constructs filtered document IDs for nested field documents
 * from these top k parent documents. Using these document IDs, it executes an exact nearest neighbor search
 * with a k value of Integer.MAX_VALUE, which provides scores for all specified nested field documents.
 */
@Builder
public class ExpandNestedDocsQuery extends Query {
    final private InternalNestedKnnVectorQuery internalNestedKnnVectorQuery;
    final private QueryUtils queryUtils;
    // Number of parent candidates to retain for rescoring, or RescoreContext.NO_RESCORE_NEEDED when rescore is off.
    @Builder.Default
    final private int rescoreK = RescoreContext.NO_RESCORE_NEEDED;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Query docAndScoreQuery = internalNestedKnnVectorQuery.knnRewrite(searcher);
        Weight weight = docAndScoreQuery.createWeight(searcher, scoreMode, boost);
        IndexReader reader = searcher.getIndexReader();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<Map<Integer, Float>> perLeafResults;
        perLeafResults = queryUtils.doSearch(searcher, leafReaderContexts, weight);
        // Built once and shared by both the rescore and the expansion passes to avoid rewriting the filter twice.
        final Weight filterWeight = getFilterWeight(searcher);
        if (rescoreK != RescoreContext.NO_RESCORE_NEEDED) {
            // Rescore the oversampled parent candidates at full precision and reduce to the top k parents
            // before expanding all of their child documents below.
            perLeafResults = rescore(searcher, leafReaderContexts, perLeafResults, filterWeight);
        }
        TopDocs[] topDocs = retrieveAll(searcher, leafReaderContexts, perLeafResults, filterWeight);
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

    /**
     * Rescores the oversampled parent candidates at full precision and reduces them to the top k parents.
     *
     * <p>For each leaf, the candidate parents' child documents are expanded to their sibling set and a
     * diversified, full-precision exact search selects the best-scoring child per parent. Segment-local doc
     * ids are rebased to global ids so results can be merged across leaves and reduced to the top k parents.
     * The surviving parents are then redistributed back into per-leaf, segment-local maps for the subsequent
     * {@link #retrieveAll} expansion, which returns every child document of these top k parents.
     *
     * <p>Per-leaf correctness relies on the invariant {@code luceneK >= rescoreK} established in
     * {@link org.opensearch.knn.index.query.KNNQueryFactory} (luceneK is defined as {@code max(rescoreK, efSearch)}).
     * The approximate pass merges leaves down to {@code rescoreK} candidates total (see
     * {@link OSDiversifyingChildrenFloatKnnVectorQuery#mergeLeafResults}), so each leaf holds at most
     * {@code rescoreK <= luceneK} candidate parents. The diversified exact search here collects up to
     * {@code luceneK} best-per-parent results, so it can never drop a candidate parent before the global
     * top-k merge below. If that invariant is ever broken this becomes a silent per-leaf truncation.
     *
     * @param indexSearcher the index searcher
     * @param leafReaderContexts the leaf reader contexts
     * @param perLeafResults per-leaf maps of the oversampled parent candidate child doc ids to their scores
     * @param filterWeight the pre-built filter weight, or null when the query has no filter
     * @return per-leaf maps containing only the surviving top k parents' child doc ids and rescored scores
     * @throws IOException on read error
     */
    private List<Map<Integer, Float>> rescore(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final List<Map<Integer, Float>> perLeafResults,
        final Weight filterWeight
    ) throws IOException {
        final List<Callable<TopDocs>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            final LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            final int finalI = i;
            rescoreTasks.add(() -> {
                final Bits queryFilter = queryUtils.createBits(leafReaderContext, filterWeight);
                final DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
                    leafReaderContext,
                    perLeafResults.get(finalI).keySet(),
                    internalNestedKnnVectorQuery.getParentFilter(),
                    queryFilter
                );
                // Diversified, full-precision exact search returns the best child per parent.
                final TopDocs topDocs = internalNestedKnnVectorQuery.knnRescoreSearch(leafReaderContext, allSiblings);
                // Rebase from segment-local to global doc ids so results can be merged across leaves.
                for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                    scoreDoc.doc = scoreDoc.doc + leafReaderContext.docBase;
                }
                return topDocs;
            });
        }
        final TopDocs[] rescored = indexSearcher.getTaskExecutor().invokeAll(rescoreTasks).toArray(TopDocs[]::new);

        // Reduce across all leaves to the top k parents.
        final int k = internalNestedKnnVectorQuery.getK();
        final TopDocs topKParents = TopDocs.merge(k, rescored);

        // Redistribute the surviving parents back to per-leaf, segment-local doc ids for the expansion step.
        final List<Map<Integer, Float>> survivingPerLeaf = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < leafReaderContexts.size(); i++) {
            survivingPerLeaf.add(new HashMap<>());
        }
        for (ScoreDoc scoreDoc : topKParents.scoreDocs) {
            final int leafIndex = ReaderUtil.subIndex(scoreDoc.doc, leafReaderContexts);
            final LeafReaderContext leafReaderContext = leafReaderContexts.get(leafIndex);
            survivingPerLeaf.get(leafIndex).put(scoreDoc.doc - leafReaderContext.docBase, scoreDoc.score);
        }
        return survivingPerLeaf;
    }

    private TopDocs[] retrieveAll(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final List<Map<Integer, Float>> perLeafResults,
        final Weight filterWeight
    ) throws IOException {
        // Construct query
        List<Callable<TopDocs>> nestedQueryTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            nestedQueryTasks.add(() -> {
                Bits queryFilter = queryUtils.createBits(leafReaderContext, filterWeight);
                DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
                    leafReaderContext,
                    perLeafResults.get(finalI).keySet(),
                    internalNestedKnnVectorQuery.getParentFilter(),
                    queryFilter
                );
                TopDocs topDocs = internalNestedKnnVectorQuery.knnExactSearch(leafReaderContext, allSiblings);
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
        return rescoreK == other.rescoreK && internalNestedKnnVectorQuery.equals(other.internalNestedKnnVectorQuery);
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
