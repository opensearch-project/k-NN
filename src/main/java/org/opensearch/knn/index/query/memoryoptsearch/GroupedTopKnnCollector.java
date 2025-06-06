/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/**
 * A {@code KnnCollector} implementation designed for nested use cases, utilizing {@link GroupedScoreMinHeap}
 * to track the highest scoring child vector per parent ID.
 * <p>
 * When multiple child vectors belong to the same parent ID were given, the collector retains the parent ID and updates
 * the stored score to the higher of the two.
 * <p>
 * For more details, refer to the {@link GroupedScoreMinHeap} JavaDoc.
 */

public class GroupedTopKnnCollector extends AbstractKnnCollector {

    private final GroupedScoreMinHeap groupedMinHeap;

    /**
     * @param k the number of neighbors to collect
     * @param visitLimit how many vector nodes the results are allowed to visit
     * @param searchStrategy the search strategy to use
     */
    public GroupedTopKnnCollector(int k, int visitLimit, KnnSearchStrategy searchStrategy, final BitSetParentIdGrouper docIdGrouper) {
        super(k, visitLimit, searchStrategy);
        this.groupedMinHeap = new GroupedScoreMinHeap(k, docIdGrouper);
    }

    @Override
    public boolean collect(int childDocId, float similarity) {
        return groupedMinHeap.insertWithOverflow(childDocId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
        if (groupedMinHeap.size() > k()) {
            return groupedMinHeap.getMinScore();
        }
        return Float.NEGATIVE_INFINITY;
    }

    @Override
    public TopDocs topDocs() {
        final ScoreDoc[] scoreDocs = new ScoreDoc[groupedMinHeap.size()];
        for (int i = 0; i < groupedMinHeap.size(); i++) {
            scoreDocs[i] = new ScoreDoc(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY);
        }
        groupedMinHeap.orderResultsInDesc(scoreDocs);
        final TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), scoreDocs);
    }

    @Override
    public int numCollected() {
        return groupedMinHeap.size();
    }

    @Override
    public String toString() {
        return "FaissGroupedTopKnnCollector[k=" + k() + ", size=" + groupedMinHeap.size() + "]";
    }
}
