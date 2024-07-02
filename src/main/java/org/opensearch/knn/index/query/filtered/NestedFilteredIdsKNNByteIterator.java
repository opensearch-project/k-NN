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
public class NestedFilteredIdsKNNByteIterator extends FilteredIdsKNNByteIterator {
    private final BitSet parentBitSet;

    public NestedFilteredIdsKNNByteIterator(
        final BitSet filterIdsArray,
        final byte[] queryVector,
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
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentBitSet.nextSetBit(docId);
        int bestChild = -1;

        while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
            binaryDocValues.advance(docId);
            float score = computeScore();
            if (score > currentScore) {
                bestChild = docId;
                currentScore = score;
            }
            docId = bitSetIterator.nextDoc();
        }

        return bestChild;
    }
}
