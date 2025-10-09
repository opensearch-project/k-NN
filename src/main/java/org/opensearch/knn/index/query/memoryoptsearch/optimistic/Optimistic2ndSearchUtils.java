/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch.optimistic;

import lombok.experimental.UtilityClass;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.hnsw.FloatHeap;
import org.opensearch.knn.index.query.PerLeafResult;

import java.util.List;

@UtilityClass
public class Optimistic2ndSearchUtils {
    /**
     * Returns the <i>k</i>-th largest score across a collection of per-leaf search results,
     * as if all scores were merged and globally sorted in descending order.
     * <p>
     * This utility is typically used to determine the global score threshold
     * corresponding to the top-<i>k</i> results when combining partial {@code TopDocs}
     * from multiple segments or shards.
     * <p>
     * The method does not perform a full global sort of all scores; it only identifies
     * the score value that would occupy the <i>k</i>-th position in the merged ranking.
     *
     * @param results       a list of {@link PerLeafResult} objects, each containing scores
     *                      collected from an individual segment or shard
     * @param k             the rank (1-based) of the desired score, e.g., {@code k = 10}
     *                      returns the 10th highest score overall
     * @param totalResults  the total number of results across all {@code results};
     *                      used for boundary checks or optimizations
     * @return the score value that would appear at position {@code k} if all scores
     *         were globally sorted in descending order
     * @throws IllegalArgumentException if {@code k} is less than 1 or greater than {@code totalResults}
     */
    public static float findKthLargestScore(final List<PerLeafResult> results, final int k, final int totalResults) {
        if (totalResults <= 0) {
            throw new IllegalArgumentException("Total results must be greater than zero, got=" + totalResults);
        }
        if (k <= 0) {
            throw new IllegalArgumentException("K must be greater than zero, got=" + k);
        }
        if (k > totalResults) {
            throw new IllegalArgumentException("K must be less than total results, got=" + k + ", totalResults=" + totalResults);
        }

        // If fewer than k scores, return the minimum score
        if (totalResults <= k) {
            float min = Float.MAX_VALUE;
            for (final PerLeafResult result : results) {
                for (final ScoreDoc scoreDoc : result.getResult().scoreDocs) {
                    if (scoreDoc.score < min) {
                        min = scoreDoc.score;
                    }
                }
            }
            return min;
        }

        // Use a min-heap to track the top-k largest values.
        // Since each PerLeafResult is already sorted in descending order by score, we push larger values first,
        // allowing the heap to fill quickly and skip most of the remaining elements.
        // This makes the practical complexity close to O(N + log K), as heap operations occur infrequently once saturated.
        final FloatHeap floatHeap = new FloatHeap(k);
        final int[] indices = new int[results.size()];
        // Maximum loop count is totalResults * #segments. Result size of segment < totalResults, therefore the upper bound (e.g. maxI)
        // becomes totalResults * #segments. Having this limit to prevent infinite loop.
        for (int i = 0, visited = 0, maxI = totalResults * results.size(); visited < totalResults && i < maxI; ++i) {
            final int resultIndex = i % indices.length;
            final int scoreIndex = indices[resultIndex];
            final ScoreDoc[] scoreDocs = results.get(resultIndex).getResult().scoreDocs;
            if (scoreIndex < scoreDocs.length) {
                floatHeap.offer(scoreDocs[scoreIndex].score);
                ++visited;
                indices[resultIndex] = scoreIndex + 1;
            }
        }

        return floatHeap.peek();
    }
}
