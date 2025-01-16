/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import java.util.HashMap;
import java.util.Map;

public class GroupedDistanceMaxHeap implements AbstractDistanceMaxHeap {
    private int k;
    private int maxK;
    private final HeapNode[] heap;
    private final Map<Integer, Entity> bestChildDistanceTracker;
    private final DocIdGrouper docIdGrouper;

    public GroupedDistanceMaxHeap(int maxSize, DocIdGrouper docIdGrouper) {
        this.docIdGrouper = docIdGrouper;
        bestChildDistanceTracker = new HashMap<>(maxSize, 0.65f);

        final int heapSize;

        if (maxSize == 0) {
            // We allocate 1 extra to avoid if statement in top()
            heapSize = 2;
        } else {
            // NOTE: we add +1 because all access to heap is
            // 1-based not 0-based. heap[0] is unused.
            heapSize = maxSize + 1;
        }

        this.heap = new HeapNode[heapSize];

        for (int i = 1; i < heapSize; i++) {
            heap[i] = new HeapNode(i, new Entity(INVALID_DOC_ID, INVALID_DOC_ID, Float.MAX_VALUE));
        }

        this.k = 0;
        this.maxK = heap.length - 1;
    }

    public void insertWithOverflow(int childId, float distance) {
        if (distance >= heap[1].entity.distance && k == maxK) {
            return;
        }

        final int groupId = docIdGrouper.getGroupId(childId);
        final Entity child = bestChildDistanceTracker.get(groupId);

        if (child == null) {
            if (k == maxK) {
                bestChildDistanceTracker.remove(heap[1].entity.groupId);
                bestChildDistanceTracker.put(groupId, heap[1].entity);
                heap[1].entity.updateGroup(groupId, childId, distance);
                downHeap(1);
            } else {
                final int index = k + 1;
                heap[index].entity.updateGroup(groupId, childId, distance);
                bestChildDistanceTracker.put(groupId, heap[index].entity);
                k = index;
                upHeap(index);
            }
        } else if (distance < child.distance) {
            child.updateChild(childId, distance);
            downHeap(child.nodeIndex);
        }
    }

    @Override
    public void orderResults(DocIdAndDistance[] results) {
        assert(results.length >= maxK);
        // K is pointing to the last valid element in the heap array. Hence, it represents the number of elements in the heap.
        // Ex: k=3 indicates that there are three elements in heap[1] - heap[3] (inclusive, 1-based index)
        // Minus 1 to convert 1-based index to 0-based index.
        int i = k - 1;
        while (i >= 0 && pop(results[i])) {
            --i;
        }
    }

    private void upHeap(int origPos) {
        int i = origPos;
        // save bottom node
        final float targetDistance = heap[i].entity.distance;
        int j = i >>> 1;
        while (j > 0 && targetDistance > heap[j].entity.distance) {
            heap[i].exchangeEntity(heap[j]);  // shift parents down
            i = j;
            j = j >>> 1;
        }
    }

    private boolean pop(DocIdAndDistance docIdAndDistance) {
        while (k > 0) {
            // We must use childId instead of group id.
            final int id = heap[1].entity.childId;
            final float distance = heap[1].entity.distance;
            heap[1].exchangeEntity(heap[k]); // move last to first
            k--;
            downHeap(1); // adjust heap

            if (id != INVALID_DOC_ID) {
                docIdAndDistance.id = id;
                docIdAndDistance.distance = distance;
                return true;
            }
        }

        return false;
    }

    private void downHeap(int i) {
        float targetDistance = heap[i].entity.distance;

        int j = i << 1; // find bigger child
        int k = (i << 1) + 1;
        if (k <= this.k && heap[k].entity.distance > heap[j].entity.distance) {
            j = k;
        }
        while (j <= this.k && heap[j].entity.distance > targetDistance) {
            heap[i].exchangeEntity(heap[j]);  // shift up child

            i = j;
            j = i << 1;
            k = j + 1;
            if (k <= this.k && heap[k].entity.distance > heap[j].entity.distance) {
                j = k;
            }
        }
    }

    private static class Entity {
        int groupId;
        int childId;
        float distance;
        int nodeIndex;

        public Entity(int groupId, int childId, float distance) {
            updateGroup(groupId, childId, distance);
        }

        public void updateGroup(int groupId, int childId, float distance) {
            this.groupId = groupId;
            this.childId = childId;
            this.distance = distance;
        }

        public void updateChild(int childId, float distance) {
            this.childId = childId;
            this.distance = distance;
        }
    }

    private static class HeapNode {
        int index;
        Entity entity;

        public HeapNode(int index, Entity entity) {
            this.index = index;
            this.entity = entity;
            this.entity.nodeIndex = index;
        }

        public Entity resetEntity(Entity newEntity) {
            Entity oldEntity = entity;
            entity = newEntity;
            entity.nodeIndex = index;
            return oldEntity;
        }

        public void exchangeEntity(HeapNode other) {
            final Entity newEntity = other.resetEntity(entity);
            resetEntity(newEntity);
        }
    }
}
