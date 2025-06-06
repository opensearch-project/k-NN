/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.ScoreDoc;

import java.util.HashMap;
import java.util.Map;

/**
 * A min-heap based on score, designed with a grouper and bounded capacity. This structure is tailored for scenarios where a document
 * has multiple child documents, each associated with a vector field. The heap tracks the minimum competitive vector found so far for each
 * parent document relative to a query vector. Specifically, it maintains the child vector that is closest to the query vector among
 * all children within the same parent document. The conversion from child IDs to group IDs is handled by the provided grouper.
 * <p>
 *
 * For example, if the heap already contains a group ID `13` with a score of `0.5`, and a new child with a score of `1.2` is added,
 * the heap updates the score for group ID 13 from 0.5 to 1.2 while maintaining the heap's contract. If the group ID is
 * encountered for the first time, a new entry `[group ID, 1.2]` is added.
 */
public class GroupedScoreMinHeap {
    private static final int INVALID_DOC_ID = -1;

    // Pointing to the last valid element in heap array.
    private int k;
    // Maximum value k can reach.
    private int maxK;
    // Internal heap array, 1-based for easier node index calculation.
    private final HeapNode[] heap;
    // Mapping parent id to entity having the best child id found so far. (e.g. having the maximum score)
    private final Map<Integer, Entity> bestChildScoreTracker;
    // Id grouper accepting child id and returns a group id.
    private final BitSetParentIdGrouper docIdGrouper;

    public GroupedScoreMinHeap(final int maxSize, final BitSetParentIdGrouper docIdGrouper) {
        this.docIdGrouper = docIdGrouper;
        // This hash map will be visited per each heap addition operation, so having 0.65 to have a sufficient space to minimize collision.
        this.bestChildScoreTracker = new HashMap<>(maxSize, 0.65f);

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

    public boolean insertWithOverflow(int childId, float score) {
        // If it's full, and return immediately when the given score is less competitive than the min score found so far.
        if (k == maxK && score <= heap[1].entity.score) {
            return false;
        }

        // Retrieve groupId (e.g. parentId) and its best child.
        final int groupId = docIdGrouper.getGroupId(childId);
        final Entity child = bestChildScoreTracker.get(groupId);

        if (child == null) {
            // This is the first time for the group id.
            if (k == maxK) {
                // It's full, replacing the top
                bestChildScoreTracker.remove(heap[1].entity.groupId);
                bestChildScoreTracker.put(groupId, heap[1].entity);
                heap[1].entity.updateGroup(groupId, childId, score);
                downHeap(1);
            } else {
                // Append it to the heap
                final int index = k + 1;
                heap[index].entity.updateGroup(groupId, childId, score);
                bestChildScoreTracker.put(groupId, heap[index].entity);
                k = index;
                upHeap(index);
            }

            return true;
        } else if (score > child.score) {
            // Update the best child with the current one.
            child.updateChild(childId, score);
            downHeap(child.nodeIndex);

            return true;
        }

        return false;
    }

    public int size() {
        return bestChildScoreTracker.size();
    }

    public float getMinScore() {
        return heap[1].entity.score;
    }

    public void orderResultsInDesc(final ScoreDoc[] results) {
        assert (results.length >= size());
        int i = size() - 1;
        while (i >= 0 && pop(results[i])) {
            --i;
        }
    }

    private void upHeap(int origPos) {
        int nodeId = origPos;
        // save bottom node
        final float targetScore = heap[nodeId].entity.score;
        int parentId = nodeId >>> 1;
        while (parentId > 0 && targetScore < heap[parentId].entity.score) {
            heap[nodeId].exchangeEntity(heap[parentId]);  // shift parents down
            nodeId = parentId;
            parentId = parentId >>> 1;
        }
    }

    private boolean pop(final ScoreDoc scoreDoc) {
        while (k > 0) {
            // We must use childId instead of group id.
            final int childDocId = heap[1].entity.childDocId;
            final float score = heap[1].entity.score;
            heap[1].exchangeEntity(heap[k]); // move last to first
            k--;
            downHeap(1); // adjust heap

            if (childDocId != INVALID_DOC_ID) {
                scoreDoc.doc = childDocId;
                scoreDoc.score = score;
                return true;
            }
        }

        return false;
    }

    private void downHeap(int nodeId) {
        final float targetScore = heap[nodeId].entity.score;

        int smallerChild = nodeId << 1; // find smaller child
        int rightChild = (nodeId << 1) + 1;
        if (rightChild <= k && heap[rightChild].entity.score < heap[smallerChild].entity.score) {
            smallerChild = rightChild;
        }
        while (smallerChild <= k && heap[smallerChild].entity.score < targetScore) {
            heap[nodeId].exchangeEntity(heap[smallerChild]);  // shift up child

            nodeId = smallerChild;
            smallerChild = nodeId << 1;
            rightChild = (nodeId << 1) + 1;
            if (rightChild <= k && heap[rightChild].entity.score < heap[smallerChild].entity.score) {
                smallerChild = rightChild;
            }
        }
    }

    private static class Entity {
        int groupId;
        int childDocId;
        float score;
        int nodeIndex;

        public Entity(int groupId, int childId, float score) {
            updateGroup(groupId, childId, score);
        }

        public void updateGroup(int groupId, int childId, float score) {
            this.groupId = groupId;
            this.childDocId = childId;
            this.score = score;
        }

        public void updateChild(int childId, float score) {
            this.childDocId = childId;
            this.score = score;
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
