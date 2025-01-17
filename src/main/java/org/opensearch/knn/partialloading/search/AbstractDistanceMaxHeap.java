/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

/**
 * A max-heap based on distance with a fixed size.
 * If there is still room for elements, it will retain them. However, once it reaches capacity, it will evict the farthest vector to make
 * space for a new one then add it.
 */
public interface AbstractDistanceMaxHeap {
    /**
     * Add a new pair of id and distance to heap.
     * If there is still room for elements, it will retain them. However, once it reaches capacity, it will evict the farthest vector to
     * make space for a new one then add it.
     *
     * @param id ID of a new element. While there is no strict restriction on the type of ID, a Lucene document ID is typically expected.
     * @param distance The distance between a vector and the query vector.
     */
    void insertWithOverflow(int id, float distance);

    /**
     * Flushes all retained elements into the given results, ordered by distance in increasing order.
     *
     * @param results A results array with a length that is at least equal to the configured capacity of this heap. It must be non-null
     *                and contain only non-null instances.
     */
    void orderResults(IdAndDistance[] results);
}
