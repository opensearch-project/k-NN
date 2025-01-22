/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.PlainDistanceMaxHeap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

public class PlainDistanceMaxHeapTests extends KNNTestCase {
    public void testPopMin() {
        // Prepare data set.
        final int k = 100;
        final int numDocs = k * 5;
        final Map<Integer, Float> distances = new HashMap<>();
        final List<Integer> sortedDocIds = generateTestData(numDocs, distances);

        // Start inserting
        List<Integer> docIds = new ArrayList<>(distances.keySet());
        Collections.shuffle(docIds);
        PlainDistanceMaxHeap heap = new PlainDistanceMaxHeap(k);
        for (int docId : docIds) {
            heap.insertWithOverflow(docId, distances.get(docId));
        }

        // Pop min 5 times
        IdAndDistance minDoc = new IdAndDistance();
        for (int i = 0; i < 5; ++i) {
            heap.popMin(minDoc);
            assertEquals((int) sortedDocIds.get(i), minDoc.id);
            assertEquals(distances.get(sortedDocIds.get(i)), minDoc.distance, 1e-6);
        }

        // Results validation
        IdAndDistance[] results = new IdAndDistance[k];
        for (int i = 0; i < k; ++i) {
            results[i] = new IdAndDistance(IdAndDistance.INVALID_DOC_ID, 0);
        }

        // If I add a pair having more close distance, then we must get it in the next popMin
        int newDocId = numDocs;
        float newMinDistance = distances.get(sortedDocIds.get(0)) - 1;
        heap.insertWithOverflow(newDocId, newMinDistance);

        heap.popMin(minDoc);
        assertEquals(newDocId, minDoc.id);
        assertEquals(newMinDistance, minDoc.distance, 1e-6);

        // If we pour new more data into the heap, we should still be able to get the correct answer.
        List<Integer> newResultDocIds = new ArrayList<>();
        List<Float> newResultDistances = new ArrayList<>();

        --newDocId;
        --newMinDistance;
        for (int i = 0; i < k; ++i) {
            newResultDocIds.add(newDocId);
            newResultDistances.add(newMinDistance);
            heap.insertWithOverflow(newDocId, newMinDistance);
        }
        Collections.reverse(newResultDocIds);
        Collections.reverse(newResultDistances);

        // Start validation
        for (int i = 0; i < k; ++i) {
            heap.popMin(minDoc);
            assertEquals((int) newResultDocIds.get(i), minDoc.id);
            assertEquals(newResultDistances.get(i), minDoc.distance, 1e-6);
        }
    }

    public void testInsertCaseWhereNumDocsGreaterThanK() {
        // Prepare data set.
        final int k = 100;
        final int numDocs = k * 5;
        final Map<Integer, Float> distances = new HashMap<>();
        final List<Integer> sortedDocIds = generateTestData(numDocs, distances);

        // Start inserting
        List<Integer> docIds = new ArrayList<>(distances.keySet());
        Collections.shuffle(docIds);
        PlainDistanceMaxHeap heap = new PlainDistanceMaxHeap(k);
        for (int docId : docIds) {
            heap.insertWithOverflow(docId, distances.get(docId));
        }

        // Results validation
        IdAndDistance[] results = new IdAndDistance[k];
        for (int i = 0; i < k; ++i) {
            results[i] = new IdAndDistance(IdAndDistance.INVALID_DOC_ID, 0);
        }
        heap.orderResults(results);
        assertNotEquals(IdAndDistance.INVALID_DOC_ID, results[results.length - 1].id);

        for (int i = 0; i < k; ++i) {
            assertEquals((int) sortedDocIds.get(i), results[i].id);
            assertEquals(distances.get(sortedDocIds.get(i)), results[i].distance, 1e-6);
        }
    }

    private List<Integer> generateTestData(int numDocs, Map<Integer, Float> distances) {
        // Min heap
        PriorityQueue<Integer> expectedHeap = new PriorityQueue<>((id1, id2) -> Float.compare(distances.get(id1), distances.get(id2)));

        for (int i = 0; i < numDocs; ++i) {
            final float distance = ThreadLocalRandom.current().nextFloat();
            distances.put(i, distance);
            expectedHeap.add(i);
        }

        List<Integer> sortedDocIds = new ArrayList<>();
        while (!expectedHeap.isEmpty()) {
            sortedDocIds.add(expectedHeap.poll());
        }

        return sortedDocIds;
    }

    public void testInsertCaseWhereNumDocsLessThanK() {
        // Prepare data set.
        final int k = 100;
        final int numDocs = k / 2;
        final Map<Integer, Float> distances = new HashMap<>();
        final List<Integer> sortedDocIds = generateTestData(numDocs, distances);

        // Start inserting
        final List<Integer> docIds = new ArrayList<>(distances.keySet());
        Collections.shuffle(docIds);
        PlainDistanceMaxHeap heap = new PlainDistanceMaxHeap(k);
        for (int docId : docIds) {
            heap.insertWithOverflow(docId, distances.get(docId));
        }

        // Results validation
        IdAndDistance[] results = new IdAndDistance[k];
        for (int i = 0; i < k; ++i) {
            results[i] = new IdAndDistance(IdAndDistance.INVALID_DOC_ID, 0);
        }
        heap.orderResults(results);

        // We should have valid values for results[0:len(docs)-1]
        for (int i = 0; i < distances.size(); ++i) {
            assertNotEquals(IdAndDistance.INVALID_DOC_ID, results[i].id);
        }
        // We should not have valid values for results[len(docs):]
        for (int i = distances.size(); i < k; ++i) {
            assertEquals(IdAndDistance.INVALID_DOC_ID, results[i].id);
        }

        // Validate we got the correct values.
        for (int i = 0; i < distances.size(); ++i) {
            assertEquals((int) sortedDocIds.get(i), results[i].id);
            assertEquals(distances.get(sortedDocIds.get(i)), results[i].distance, 1e-6);
        }
    }
}
