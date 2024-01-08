/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.filtered;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score. However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedFilteredIdsKNNIterator extends FilteredIdsKNNIterator {
    private final BitSet parentBitSet;

    public NestedFilteredIdsKNNIterator(
        final int[] filterIdsArray,
        final float[] queryVector,
        final BinaryDocValues values,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) {
        super(filterIdsArray, queryVector, values, spaceType);
        this.parentBitSet = parentBitSet;
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public int nextDoc() throws IOException {
        if (currentPos >= filterIdsArray.length) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentBitSet.nextSetBit(filterIdsArray[currentPos]);
        int bestChild = -1;
        while (currentPos < filterIdsArray.length && filterIdsArray[currentPos] < currentParent) {
            binaryDocValues.advance(filterIdsArray[currentPos]);
            float score = computeScore();
            if (score > currentScore) {
                bestChild = filterIdsArray[currentPos];
                currentScore = score;
            }
            currentPos++;
        }

        return bestChild;
    }
}
