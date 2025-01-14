/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import lombok.RequiredArgsConstructor;
import org.opensearch.knn.index.query.KNNQueryResult;

@RequiredArgsConstructor
public class HeapResultsCollector extends ResultsCollector {
    private final DistanceMaxHeap distanceMaxHeap;
    private KNNQueryResult[] results;

    @Override
    public void addResult(int docId, float distance) {
        distanceMaxHeap.insertWithOverflow(docId, distance);
    }

    @Override
    public KNNQueryResult[] getResults() {
        if (results != null) {
            return results;
        }

        final DocIdAndDistance[] heapArray = distanceMaxHeap.getHeapArray();
        results = new KNNQueryResult[distanceMaxHeap.size()];
        for (int i = DistanceMaxHeap.ONE_BASE; i < heapArray.length; ++i) {
            results[i - 1] = new KNNQueryResult(heapArray[i].id, heapArray[i].distance);
        }
        return results;
    }
}
