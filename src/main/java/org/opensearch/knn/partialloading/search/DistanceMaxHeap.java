/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class DistanceMaxHeap implements Iterable<DocIdAndDistance> {
    public static final int ONE_BASE = 1;

    private int k;
    private int numValidElems;
    private final int maxK;
    private final DocIdAndDistance[] heap;

    public DistanceMaxHeap(int maxSize) {
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
        this.heap = new DocIdAndDistance[heapSize];
        this.maxK = heapSize - 1;

        for (int i = 1; i < heapSize; i++) {
            heap[i] = new DocIdAndDistance(0, Float.MAX_VALUE);
        }

        this.k = 0;
        this.numValidElems = 0;
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
            if (heap[i].id != -1 && heap[i].distance < minDistance) {
                minIdx = i;
                minDistance = heap[i].distance;
            }
        }

        return minIdx;
    }

    public final void popMin(DocIdAndDistance minIad) {
        final int minIdx = findMinimumIndex();
        minIad.id = heap[minIdx].id;
        minIad.distance = heap[minIdx].distance;
        heap[minIdx].id = -1;
        --numValidElems;
    }

    public void insertWithOverflow(int id, float distance) {
        if (k == maxK) {
            if (distance >= heap[1].distance) {
                return;
            }
            if (heap[1].id == -1) {
                ++numValidElems;
            }

            heap[1].id = id;
            heap[1].distance = distance;
            updateTop();
        } else {
            add(id, distance);
            ++numValidElems;
        }
    }

    private static boolean greaterThan(float a, float b) {
        return a > b;
    }

    public final DocIdAndDistance top() {
        return heap[1];
    }

    public final DocIdAndDistance pop() {
        if (k > 0) {
            DocIdAndDistance result = heap[1]; // save first value
            heap[1] = heap[k]; // move last to first
            k--;
            downHeap(1); // adjust heap
            --numValidElems;
            return result;
        } else {
            return null;
        }
    }

    private DocIdAndDistance updateTop() {
        downHeap(1);
        return heap[1];
    }

    public final int size() {
        return numValidElems;
    }

    public boolean isEmpty() {
        return numValidElems <= 0;
    }

    public boolean isFull() {
        return k == maxK;
    }

    private void upHeap(int origPos) {
        int i = origPos;
        DocIdAndDistance node = heap[i]; // save bottom node
        int j = i >>> 1;
        while (j > 0 && greaterThan(node.distance, heap[j].distance)) {
            heap[i] = heap[j]; // shift parents down
            i = j;
            j = j >>> 1;
        }
        heap[i] = node; // install saved node
    }

    private void downHeap(int i) {
        DocIdAndDistance node = heap[i]; // save top node
        int j = i << 1; // find bigger child
        int k = (i << 1) + 1;
        if (k <= this.k && greaterThan(heap[k].distance, heap[j].distance)) {
            j = k;
        }
        while (j <= this.k && greaterThan(heap[j].distance, node.distance)) {
            heap[i] = heap[j]; // shift up child
            i = j;
            j = i << 1;
            k = j + 1;
            if (k <= this.k && greaterThan(heap[k].distance, heap[j].distance)) {
                j = k;
            }
        }
        heap[i] = node; // install saved node
    }

    public DocIdAndDistance[] getHeapArray() {
        return heap;
    }

    @Override
    public Iterator<DocIdAndDistance> iterator() {
        return new Iterator<>() {
            int i = 1;

            @Override
            public boolean hasNext() {
                return i <= k;
            }

            @Override
            public DocIdAndDistance next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return heap[i++];
            }
        };
    }
}
