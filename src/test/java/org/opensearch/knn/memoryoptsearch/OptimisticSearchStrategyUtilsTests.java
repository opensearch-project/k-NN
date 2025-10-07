/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.junit.Test;
import org.opensearch.knn.index.query.PerLeafResult;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.opensearch.knn.index.query.memoryoptsearch.optimistic.OptimisticSearchStrategyUtils.findKthLargestScore;

public class OptimisticSearchStrategyUtilsTests {
    /**
     * Helper method to create a {@link PerLeafResult} with given scores.
     */
    private static PerLeafResult perLeaf(float... scores) {
        ScoreDoc[] scoreDocs = new ScoreDoc[scores.length];
        for (int i = 0; i < scores.length; i++) {
            scoreDocs[i] = new ScoreDoc(i, scores[i]);
        }
        TopDocs topDocs = new TopDocs(new TotalHits(scores.length, TotalHits.Relation.EQUAL_TO), scoreDocs);
        return new PerLeafResult(null, 0, topDocs, PerLeafResult.SearchMode.APPROXIMATE_SEARCH);
    }

    @Test
    public void testSingleSegmentSimple() {
        List<PerLeafResult> results = List.of(perLeaf(9.5f, 8.2f, 7.1f, 5.0f));
        float score = findKthLargestScore(results, 2, 4);
        assertEquals(8.2f, score, 1e-6);
    }

    @Test
    public void testMultiSegmentMerge() {
        List<PerLeafResult> results = List.of(perLeaf(9.0f, 3.0f), perLeaf(8.5f, 7.2f, 4.4f), perLeaf(6.8f));
        // All scores combined: [9.0, 8.5, 7.2, 6.8, 4.4, 3.0]
        // .................................^-------- This is what we're looking for
        float score = findKthLargestScore(results, 3, 6);
        assertEquals(7.2f, score, 1e-6);
    }

    @Test
    public void testTiedScores() {
        List<PerLeafResult> results = List.of(perLeaf(9.0f, 9.0f, 8.0f), perLeaf(8.0f, 7.5f));
        // Combined sorted: [9.0, 9.0, 8.0, 8.0, 7.5]
        assertEquals(9.0f, findKthLargestScore(results, 1, 5), 1e-6);
        assertEquals(9.0f, findKthLargestScore(results, 2, 5), 1e-6);
        assertEquals(8.0f, findKthLargestScore(results, 3, 5), 1e-6);
    }

    @Test
    public void testKEqualsTotalResults() {
        List<PerLeafResult> results = List.of(perLeaf(5.0f, 6.0f), perLeaf(7.0f));
        // Combined: [7.0, 6.0, 5.0]
        float score = findKthLargestScore(results, 3, 3);
        assertEquals(5.0f, score, 1e-6);
    }

    @Test
    public void testInvalidK() {
        List<PerLeafResult> results = List.of(perLeaf(1.0f, 2.0f));
        assertThrows(IllegalArgumentException.class, () -> findKthLargestScore(results, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> findKthLargestScore(results, 3, 2));
    }

    @Test
    public void testEmptyResults() {
        List<PerLeafResult> results = new ArrayList<>();
        assertThrows(IllegalArgumentException.class, () -> findKthLargestScore(results, 1, 0));
    }

    @Test
    public void testAllSameScores() {
        List<PerLeafResult> results = List.of(perLeaf(5.0f, 5.0f), perLeaf(5.0f));
        float score = findKthLargestScore(results, 2, 3);
        assertEquals(5.0f, score, 1e-6);
    }

    @Test
    public void testLargeKMultiSegment() {
        List<PerLeafResult> results = List.of(perLeaf(10f, 9f, 8f), perLeaf(7f, 6f), perLeaf(5f, 4f));
        // Combined sorted: [10, 9, 8, 7, 6, 5, 4]
        assertEquals(4f, findKthLargestScore(results, 7, 7), 1e-6);
    }
}
