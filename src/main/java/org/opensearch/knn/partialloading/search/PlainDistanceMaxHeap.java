/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import static org.opensearch.knn.partialloading.search.IdAndDistance.INVALID_DOC_ID;

/**
 * A max-heap based on distance with a fixed capacity.
 * If there is room, it will add elements to the heap; otherwise, it will only add a new element if its distance is competitive with the
 * maximum distance found so far.
 */
public class PlainDistanceMaxHeap implements AbstractDistanceMaxHeap {
    // Pointing to the last valid element in heap array.
    private int k;
    // The number of valid elements (e.g. not popped yet) in internal heap array.
    private int numValidElems;
    // Maximum value k can reach.
    private final int maxK;
    // Internal heap array, 1-based for easier node index calculation.
    private final IdAndDistance[] heap;

    public PlainDistanceMaxHeap(int maxSize) {
        final int heapSize;

        if (maxSize == 0) {
            // We allocate 1 extra to avoid if statement in top()
            heapSize = 2;
        } else {
            // NOTE: we add +1 because all access to heap is
            // 1-based not 0-based. heap[0] is unused.
            heapSize = maxSize + 1;
        }

        // T is an unbounded type, so this unchecked cast works always.
        this.heap = new IdAndDistance[heapSize];
        this.maxK = heapSize - 1;

        for (int i = 1; i < heapSize; i++) {
            heap[i] = new IdAndDistance(0, Float.MAX_VALUE);
        }

        this.k = 0;
        this.numValidElems = 0;
    }

    /**
     * Find the minimum distance and update `minIad`.
     * Note that this pop operation does not evict the found minimum element from the heap, as outlined in the HNSW paper, to ensure that
     * the search converges correctly.
     *
     * @param minIad Minimum vector id and distance found from the heap.
     */
    public void popMin(IdAndDistance minIad) {
        final int minIdx = findMinimumIndex();
        minIad.id = heap[minIdx].id;
        minIad.distance = heap[minIdx].distance;
        heap[minIdx].id = INVALID_DOC_ID;
        --numValidElems;
    }

    @Override
    public void insertWithOverflow(int id, float distance) {
        if (k == maxK) {
            if (distance >= heap[1].distance) {
                // Did not make cut. Return immediately.
                return;
            }

            // We are replacing the top. And since it was already popped, we should increase #valid elements.
            if (heap[1].id == INVALID_DOC_ID) {
                ++numValidElems;
            }

            heap[1].id = id;
            heap[1].distance = distance;
            downHeap(1);
        } else {
            add(id, distance);
            ++numValidElems;
        }
    }

    @Override
    public void orderResults(IdAndDistance[] results) {
        int i = numValidElems - 1;
        while (i >= 0) {
            final IdAndDistance popped = pop();
            results[i].id = popped.id;
            results[i].distance = popped.distance;
            --i;
        }
    }

    private void add(int id, float distance) {
        // don't modify size until we know heap access didn't throw AIOOB.
        final int index = k + 1;
        heap[index].id = id;
        heap[index].distance = distance;
        k = index;
        upHeap(index);
    }

    private int findMinimumIndex() {
        float minDistance = Float.MAX_VALUE;
        int minIdx = -1;
        for (int i = k; i > 0; --i) {
            if (heap[i].distance < minDistance && heap[i].id != INVALID_DOC_ID) {
                minIdx = i;
                minDistance = heap[i].distance;
            }
        }

        return minIdx;
    }

    private IdAndDistance pop() {
        while (true) {
            if (k > 0) {
                IdAndDistance result = heap[1]; // save first value
                heap[1] = heap[k]; // move last to first
                k--;
                downHeap(1); // adjust heap

                if (result.id != INVALID_DOC_ID) {
                    return result;
                }
            } else {
                return null;
            }
        }
    }

    public boolean isEmpty() {
        return numValidElems <= 0;
    }

    private void upHeap(int origPos) {
        int nodeIdx = origPos;
        IdAndDistance node = heap[nodeIdx]; // save bottom node
        int parentIdx = nodeIdx >>> 1;
        while (parentIdx > 0 && node.distance > heap[parentIdx].distance) {
            heap[nodeIdx] = heap[parentIdx]; // shift parents down
            nodeIdx = parentIdx;
            parentIdx = parentIdx >>> 1;
        }
        heap[nodeIdx] = node; // install saved node
    }

    private void downHeap(int nodeIdx) {
        IdAndDistance node = heap[nodeIdx]; // save top node
        int biggestChildIdx = nodeIdx << 1; // find bigger child
        int rigntChildIdx = (nodeIdx << 1) + 1;
        if (rigntChildIdx <= this.k && heap[rigntChildIdx].distance > heap[biggestChildIdx].distance) {
            biggestChildIdx = rigntChildIdx;
        }
        while (biggestChildIdx <= this.k && heap[biggestChildIdx].distance > node.distance) {
            heap[nodeIdx] = heap[biggestChildIdx]; // shift up child
            nodeIdx = biggestChildIdx;
            biggestChildIdx = nodeIdx << 1;
            rigntChildIdx = (nodeIdx << 1) + 1;
            if (rigntChildIdx <= this.k && heap[rigntChildIdx].distance > heap[biggestChildIdx].distance) {
                biggestChildIdx = rigntChildIdx;
            }
        }
        heap[nodeIdx] = node; // install saved node
    }
}
