/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to scoreif filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 * When expandNested is true, it returns ALL child docs instead of just the best one.
 */
public class NestedBinaryVectorIdsKNNIterator extends BinaryVectorIdsKNNIterator {
    private final BitSet parentBitSet;
    private final boolean expandNested;

    public NestedBinaryVectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet,
        final boolean expandNested
    ) throws IOException {
        super(filterIdsIterator, queryVector, binaryVectorValues, spaceType);
        this.parentBitSet = parentBitSet;
        this.expandNested = expandNested;
    }

    public NestedBinaryVectorIdsKNNIterator(
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        super(null, queryVector, binaryVectorValues, spaceType);
        this.parentBitSet = parentBitSet;
        this.expandNested = false;
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * When expandNested is true, returns ALL child docs instead of just the best one.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next child doc id (best child when expandNested=false, all children when expandNested=true)
     */
    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        if (expandNested) {
            int currentParent = parentBitSet.nextSetBit(docId);
            if (currentParent != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
                currentScore = computeScore();
                int currentDocId = docId;
                docId = getNextDocId();
                return currentDocId;
            }
            docId = getNextDocId();
            return nextDoc();
        }

        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentBitSet.nextSetBit(docId);
        int bestChild = -1;

        // In order to traverse all children for given parent, we have to use docId < parentId, because,
        // kNNVectorValues will not have parent id since DocId is unique per segment. For ex: let's say for doc id 1, there is one child
        // and for doc id 5, there are three children. In that case knnVectorValues iterator will have [0, 2, 3, 4]
        // and parentBitSet will have [1,5]
        // Hence, we have to iterate till docId from knnVectorValues is less than parentId instead of till equal to parentId
        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            float score = computeScore();
            if (score > currentScore) {
                bestChild = docId;
                currentScore = score;
            }
            docId = getNextDocId();
        }

        return bestChild;
    }
}
