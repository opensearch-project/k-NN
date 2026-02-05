/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;

import java.io.IOException;

/**
 * Iterator for exact KNN search using quantized vectors with nested field support.
 * Deduplicates docs per parent and returns the best child doc with the highest score.
 */
class NestedQuantizedVectorIdsExactKNNIterator extends QuantizedVectorIdsExactKNNIterator {
    private final BitSet parentBitSet;

    public NestedQuantizedVectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final KnnVectorValues.DocIndexIterator docIndexIterator,
        final ByteVectorValues byteVectorValues,
        final byte[] quantizedQueryVector,
        final BitSet parentBitSet
    ) throws IOException {
        super(filterIdsIterator, docIndexIterator, byteVectorValues, quantizedQueryVector);
        this.parentBitSet = parentBitSet;
    }

    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }

        currentScore = Float.NEGATIVE_INFINITY;
        int currentParent = parentBitSet.nextSetBit(docId);
        int bestChild = -1;

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
