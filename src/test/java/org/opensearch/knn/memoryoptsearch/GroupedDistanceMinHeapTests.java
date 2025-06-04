/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.search.ScoreDoc;
import org.junit.Test;
import org.opensearch.knn.index.query.memoryoptsearch.BitSetParentIdGrouper;
import org.opensearch.knn.index.query.memoryoptsearch.GroupedScoreMinHeap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class GroupedDistanceMinHeapTests {
    @Test
    public void testGroupedDistanceMinHeap() {
        // Make grouper
        // - 0-3's parent doc id=4
        // - 5-99's parent doc id=100
        // - etc
        final int[] parentIds = new int[] { 4, 100, 104, 1024, 3000, 4000, 5000, 6000, 7000, 8000, 11000 };
        Map<Integer, Integer> childToParentId = new HashMap<>();
        for (int i = 0, lastParentId = -1; i < parentIds.length; i++) {
            final int parentId = parentIds[i];
            for (int childId = lastParentId + 1; childId < parentId; childId++) {
                childToParentId.put(childId, parentId);
            }
            lastParentId = parentId;
        }

        final BitSetParentIdGrouper grouper = BitSetParentIdGrouper.createGrouper(parentIds);

        // Make test data
        Map<Integer, Float> childScores = new HashMap<>();
        Map<Integer, Integer> maxChildPerParentId = new HashMap<>();
        List<Integer> childIds = new ArrayList<>(childToParentId.keySet());
        for (Map.Entry<Integer, Integer> entry : childToParentId.entrySet()) {
            final int childId = entry.getKey();
            final float score = ThreadLocalRandom.current().nextFloat();
            final float finalScore = -1000 + score * 2000;  // range: [-1000, 1000)
            childScores.put(entry.getKey(), finalScore);
            // Update max child
            maxChildPerParentId.compute(
                entry.getValue(),
                (pid, cid) -> cid == null ? childId : finalScore > childScores.get(cid) ? childId : cid
            );
        }
        List<Integer> topKParentIds = Arrays.stream(parentIds).boxed().collect(Collectors.toList());
        topKParentIds.sort(Comparator.comparing(pid -> -childScores.get(maxChildPerParentId.get(pid))));
        Collections.shuffle(childIds);

        // When data is sufficient to collect top-k
        validate(parentIds.length / 2, grouper, childIds, childScores, topKParentIds, maxChildPerParentId);
        // When data is not sufficient to collect top-k
        validate(parentIds.length * 2, grouper, childIds, childScores, topKParentIds, maxChildPerParentId);
    }

    private static void validate(
        int k,
        BitSetParentIdGrouper grouper,
        List<Integer> childIds,
        Map<Integer, Float> childScores,
        List<Integer> sortedParentIdsByScore,
        Map<Integer, Integer> minChildPerParentId
    ) {
        // Insert scores
        GroupedScoreMinHeap heap = new GroupedScoreMinHeap(k, grouper);
        for (int childId : childIds) {
            heap.insertWithOverflow(childId, childScores.get(childId));
        }

        // Get top k results
        ScoreDoc[] results = new ScoreDoc[k];
        for (int i = 0; i < k; ++i) {
            results[i] = new ScoreDoc(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY);
        }
        heap.orderResultsInDesc(results);

        // Start validation
        if (sortedParentIdsByScore.size() >= k) {
            // Collected `k` elements, all must have valid doc id.
            for (final ScoreDoc result : results) {
                assertNotEquals(Integer.MAX_VALUE, result.doc);
                assertNotEquals(Float.NEGATIVE_INFINITY, result.score);
            }
        } else {
            // Collected less than `k` elements. For the first `k` elements, must ensure all have valid doc ids.
            for (int i = 0; i < sortedParentIdsByScore.size(); ++i) {
                assertNotEquals(Integer.MAX_VALUE, results[i].doc);
            }
            // Remaining elements should have the default doc ids.
            for (int i = sortedParentIdsByScore.size(); i < results.length; ++i) {
                assertEquals(Integer.MAX_VALUE, results[i].doc);
            }
        }

        // Validate whether it has a valid child docs.
        for (int i = 0; i < Math.min(sortedParentIdsByScore.size(), k); ++i) {
            final int topParentId = sortedParentIdsByScore.get(i);
            final int expectedChildId = minChildPerParentId.get(topParentId);

            final int childId = results[i].doc;

            assertEquals(expectedChildId, childId);
            assertEquals(childScores.get(expectedChildId), results[i].score, 1e-6);
        }
    }
}
