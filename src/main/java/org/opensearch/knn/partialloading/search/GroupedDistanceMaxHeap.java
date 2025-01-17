/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.partialloading.search.IdAndDistance.INVALID_DOC_ID;

/**
 * A max-heap based on distance, designed with a grouper and bounded capacity. This structure is tailored for scenarios where a document
 * has multiple child documents, each associated with a vector field. The heap tracks the best (i.e., closest) vector found so far for
 * each parent document relative to a query vector. Specifically, it maintains the child vector that is closest to the query vector among
 * all children within the same parent document. The conversion from child IDs to group IDs is handled by the provided grouper.
 * <p>
 *
 * For example, if the heap already contains a group ID `13` with a distance of `0.5`, and a new child with a distance of `0.2` is added,
 * the heap updates the distance for group ID 13 from 0.5 to 0.2 while maintaining the heap's contract. If the group ID is
 * encountered for the first time, a new entry `[group ID, 0.2]` is added.
 */
public class GroupedDistanceMaxHeap implements AbstractDistanceMaxHeap {
    // Pointing to the last valid element in heap array.
    private int k;
    // Maximum value k can reach.
    private int maxK;
    // Internal heap array, 1-based for easier node index calculation.
    private final HeapNode[] heap;
    // Mapping parent id to entity having the best child id found so far. (e.g. having the minimum distance)
    private final Map<Integer, Entity> bestChildDistanceTracker;
    // Id grouper accepting child id and returns a group id.
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
        // If it's full, and return immediately when the given distance is less comparative the max distance found so far.
        if (distance >= heap[1].entity.distance && k == maxK) {
            return;
        }

        // Retrieve groupId (e.g. parentId) and its best child.
        final int groupId = docIdGrouper.getGroupId(childId);
        final Entity child = bestChildDistanceTracker.get(groupId);

        if (child == null) {
            // This is the first time for the group id.
            if (k == maxK) {
                // Replace the top
                bestChildDistanceTracker.remove(heap[1].entity.groupId);
                bestChildDistanceTracker.put(groupId, heap[1].entity);
                heap[1].entity.updateGroup(groupId, childId, distance);
                downHeap(1);
            } else {
                // Add it to the heap
                final int index = k + 1;
                heap[index].entity.updateGroup(groupId, childId, distance);
                bestChildDistanceTracker.put(groupId, heap[index].entity);
                k = index;
                upHeap(index);
            }
        } else if (distance < child.distance) {
            // Update the best child with the current one.
            child.updateChild(childId, distance);
            downHeap(child.nodeIndex);
        }
    }

    @Override
    public void orderResults(IdAndDistance[] results) {
        assert (results.length >= maxK);
        // K is pointing to the last valid element in the heap array. Hence, it represents the number of elements in the heap.
        // Ex: k=3 indicates that there are three elements in heap[1] - heap[3] (inclusive, 1-based index)
        // Minus 1 to convert 1-based index to 0-based index.
        int i = k - 1;
        while (i >= 0 && pop(results[i])) {
            --i;
        }
    }

    private void upHeap(int origPos) {
        int nodeId = origPos;
        // save bottom node
        final float targetDistance = heap[nodeId].entity.distance;
        int parentId = nodeId >>> 1;
        while (parentId > 0 && targetDistance > heap[parentId].entity.distance) {
            heap[nodeId].exchangeEntity(heap[parentId]);  // shift parents down
            nodeId = parentId;
            parentId = parentId >>> 1;
        }
    }

    private boolean pop(IdAndDistance idAndDistance) {
        while (k > 0) {
            // We must use childId instead of group id.
            final int id = heap[1].entity.childId;
            final float distance = heap[1].entity.distance;
            heap[1].exchangeEntity(heap[k]); // move last to first
            k--;
            downHeap(1); // adjust heap

            if (id != INVALID_DOC_ID) {
                idAndDistance.id = id;
                idAndDistance.distance = distance;
                return true;
            }
        }

        return false;
    }

    private void downHeap(int nodeId) {
        final float targetDistance = heap[nodeId].entity.distance;

        int biggestChild = nodeId << 1; // find bigger child
        int rightChild = (nodeId << 1) + 1;
        if (rightChild <= k && heap[rightChild].entity.distance > heap[biggestChild].entity.distance) {
            biggestChild = rightChild;
        }
        while (biggestChild <= k && heap[biggestChild].entity.distance > targetDistance) {
            heap[nodeId].exchangeEntity(heap[biggestChild]);  // shift up child

            nodeId = biggestChild;
            biggestChild = nodeId << 1;
            rightChild = (nodeId << 1) + 1;
            if (rightChild <= k && heap[rightChild].entity.distance > heap[biggestChild].entity.distance) {
                biggestChild = rightChild;
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
