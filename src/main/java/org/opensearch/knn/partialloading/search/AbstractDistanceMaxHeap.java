/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

public interface AbstractDistanceMaxHeap {
    public static final int INVALID_DOC_ID = -1;

    void insertWithOverflow(int id, float distance);

    void orderResults(DocIdAndDistance[] results);
}
