/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score if filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedByteVectorIdsKNNIterator extends ByteVectorIdsKNNIterator {
    private final BitSet parentBitSet;

    public NestedByteVectorIdsKNNIterator(
        final BitSet filterIdsArray,
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        super(filterIdsArray, queryVector, binaryVectorValues, spaceType);
        this.parentBitSet = parentBitSet;
    }

    NestedByteVectorIdsKNNIterator(
        final byte[] queryVector,
        final KNNBinaryVectorValues binaryVectorValues,
        final SpaceType spaceType,
        final BitSet parentBitSet
    ) throws IOException {
        super(null, queryVector, binaryVectorValues, spaceType);
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
            if (bitSetIterator != null) {
                binaryVectorValues.advance(docId);
            }
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
