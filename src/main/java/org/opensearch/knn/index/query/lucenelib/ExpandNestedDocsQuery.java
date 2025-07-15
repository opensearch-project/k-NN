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
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.AbstractProfileBreakdown;
import org.opensearch.search.profile.ContextualProfileBreakdown;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        QueryProfiler profiler = ((ContextIndexSearcher) searcher).getProfiler();
        if(profiler != null) {
            profiler.getQueryBreakdown((Query) internalNestedKnnVectorQuery);
        }
        Query docAndScoreQuery = internalNestedKnnVectorQuery.knnRewrite(searcher);
        Weight weight = docAndScoreQuery.createWeight(searcher, scoreMode, boost);
        IndexReader reader = searcher.getIndexReader();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<Map<Integer, Float>> perLeafResults;
        ContextualProfileBreakdown profile = null;
        if (profiler != null) {
            profile = profiler.getProfileBreakdown(this);
        }
        perLeafResults = queryUtils.doSearch(searcher, leafReaderContexts, weight, profile);
        TopDocs[] topDocs = retrieveAll(searcher, leafReaderContexts, perLeafResults);
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

    private TopDocs[] retrieveAll(
        final IndexSearcher indexSearcher,
        final List<LeafReaderContext> leafReaderContexts,
        final List<Map<Integer, Float>> perLeafResults
    ) throws IOException {
        // Construct query
        List<Callable<TopDocs>> nestedQueryTasks = new ArrayList<>(leafReaderContexts.size());
        Weight filterWeight = getFilterWeight(indexSearcher);
        QueryProfiler profiler = ((ContextIndexSearcher) indexSearcher).getProfiler();
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            nestedQueryTasks.add(() -> {
                Bits queryFilter;
                if (profiler != null) {
                    AbstractProfileBreakdown profile = profiler.getProfileBreakdown(this).context(leafReaderContext);
                    Timer timer = profile.getTimer(KNNQueryTimingType.BITSET_CREATION);
                    timer.start();
                    try {
                        queryFilter = queryUtils.createBits(leafReaderContext, filterWeight);
                    } finally {
                        timer.stop();
                    }
                } else {
                    queryFilter = queryUtils.createBits(leafReaderContext, filterWeight);
                }
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
        return internalNestedKnnVectorQuery.equals(other.internalNestedKnnVectorQuery);
    }

    @Override
    public int hashCode() {
        return internalNestedKnnVectorQuery.hashCode();
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
