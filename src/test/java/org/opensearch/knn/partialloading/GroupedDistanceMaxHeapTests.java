/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.DocIdGrouper;
import org.opensearch.knn.partialloading.search.GroupedDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.distance.BitSetParentIdGrouper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class GroupedDistanceMaxHeapTests extends KNNTestCase {
    public void testGroupedDistanceMaxHeap() {
        // Make grouper
        //  - 0-3's parent doc id=4
        //  - 5-99's parent doc id=100
        //  - etc
        final int[] parentIds = new int[] { 4, 100, 104, 1024, 3000, 4000, 5000, 6000, 7000, 8000, 11000 };
        Map<Integer, Integer> childToParentId = new HashMap<>();
        for (int i = 0, lastParentId = -1; i < parentIds.length; i++) {
            final int parentId = parentIds[i];
            for (int childId = lastParentId + 1; childId < parentId; childId++) {
                childToParentId.put(childId, parentId);
            }
            lastParentId = parentId;
        }

        BitSetParentIdGrouper grouper = BitSetParentIdGrouper.createGrouper(parentIds);

        // Make test data
        Map<Integer, Float> distances = new HashMap<>();
        Map<Integer, Integer> minChildPerParentId = new HashMap<>();
        List<Integer> childIds = new ArrayList<>(childToParentId.keySet());
        for (Map.Entry<Integer, Integer> entry : childToParentId.entrySet()) {
            final int childId = entry.getKey();
            final float distance = ThreadLocalRandom.current().nextFloat();
            distances.put(entry.getKey(), distance);
            // Update min child
            minChildPerParentId.compute(entry.getValue(),
                                        (pid, cid) -> cid == null ? childId : distance < distances.get(cid) ? childId : cid
            );
        }
        List<Integer> topKParentIds = Arrays.stream(parentIds).boxed().collect(Collectors.toList());
        topKParentIds.sort(Comparator.comparing(pid -> distances.get(minChildPerParentId.get(pid))));
        Collections.shuffle(childIds);

        // When data is sufficient to collect top-k
        validate(parentIds.length / 2, grouper, childIds, distances, topKParentIds, minChildPerParentId);
        // When data is not sufficient to collect top-k
        validate(parentIds.length * 2, grouper, childIds, distances, topKParentIds, minChildPerParentId);
    }

    private static void validate(
        int k,
        DocIdGrouper grouper,
        List<Integer> childIds,
        Map<Integer, Float> distances,
        List<Integer> sortedParentIdsByDistance,
        Map<Integer, Integer> minChildPerParentId
    ) {
        // Insert distances
        GroupedDistanceMaxHeap heap = new GroupedDistanceMaxHeap(k, grouper);
        for (int childId : childIds) {
            heap.insertWithOverflow(childId, distances.get(childId));
        }

        // Get top k results
        IdAndDistance[] results = new IdAndDistance[k];
        for (int i = 0; i < k; ++i) {
            results[i] = new IdAndDistance(IdAndDistance.INVALID_DOC_ID, 0);
        }
        heap.orderResults(results);

        // Start validation
        if (sortedParentIdsByDistance.size() >= k) {
            for (final IdAndDistance result : results) {
                assertNotEquals(IdAndDistance.INVALID_DOC_ID, result.id);
            }
        } else {
            for (int i = 0; i < sortedParentIdsByDistance.size(); ++i) {
                assertNotEquals(IdAndDistance.INVALID_DOC_ID, results[i].id);
            }
            for (int i = sortedParentIdsByDistance.size(); i < results.length; ++i) {
                assertEquals(IdAndDistance.INVALID_DOC_ID, results[i].id);
            }
        }

        for (int i = 0; i < Math.min(sortedParentIdsByDistance.size(), k); ++i) {
            final int topParentId = sortedParentIdsByDistance.get(i);
            final int expectedChildId = minChildPerParentId.get(topParentId);

            final int childId = results[i].id;

            assertEquals(expectedChildId, childId);
            assertEquals(distances.get(expectedChildId), results[i].distance, 1e-6);
        }
    }
}
