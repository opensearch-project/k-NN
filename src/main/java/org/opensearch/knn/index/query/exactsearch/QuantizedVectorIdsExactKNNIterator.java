/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

/**
 * Iterator for exact KNN search using pre-quantized vectors from native engines.
 * Compares quantized query vector against pre-quantized document vectors using Hamming distance.
 */
class QuantizedVectorIdsExactKNNIterator implements ExactKNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    private final byte[] quantizedQueryVector;
    protected final KnnVectorValues.DocIndexIterator knnFloatVectorValuesIndexIterator;
    protected final ByteVectorValues quantizedByteVectorValues;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int docId;

    /**
     * Creates an iterator for exact KNN search using pre-quantized vectors.
     *
     * @param filterIdsIterator optional filter to restrict search to specific documents
     * @param knnFloatVectorValuesIndexIterator iterator for document indices
     * @param quantizedByteVectorValues pre-quantized document vectors from native engine
     * @param quantizedQueryVector quantized query vector
     * @throws IOException if an I/O error occurs
     */
    public QuantizedVectorIdsExactKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final KnnVectorValues.DocIndexIterator knnFloatVectorValuesIndexIterator,
        final ByteVectorValues quantizedByteVectorValues,
        final byte[] quantizedQueryVector
    ) throws IOException {
        this.filterIdsIterator = filterIdsIterator;
        this.knnFloatVectorValuesIndexIterator = knnFloatVectorValuesIndexIterator;
        this.quantizedByteVectorValues = quantizedByteVectorValues;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        this.docId = getNextDocId();
        this.quantizedQueryVector = quantizedQueryVector;
    }

    /**
     * Advance to the next doc and update score value with score of the next doc.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next doc id
     */
    @Override
    public int nextDoc() throws IOException {

        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        currentScore = computeScore();
        int currentDocId = docId;
        docId = getNextDocId();
        return currentDocId;
    }

    @Override
    public float score() {
        return currentScore;
    }

    /**
     * Computes similarity score between quantized query and document vectors using Hamming distance.
     * <p>
     * Quantized vectors from native engines are stored using internal vector IDs (ordinals), not document IDs.
     * <p>
     * The {@link KnnVectorValues.DocIndexIterator#index()} method returns the ordinal (internal vector ID)
     * corresponding to the current document, which is then used to retrieve the quantized vector from the
     * native engine's pre-quantized storage.
     *
     * @return similarity score for the current document
     * @throws IOException if an I/O error occurs while reading vector values
     */
    protected float computeScore() throws IOException {
        final int index = knnFloatVectorValuesIndexIterator.index();
        byte[] quantizedVector = quantizedByteVectorValues.vectorValue(index);
        return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
    }

    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return knnFloatVectorValuesIndexIterator.nextDoc();
        }
        int nextDocID = this.filterIdsIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            knnFloatVectorValuesIndexIterator.advance(nextDocID);
        }
        return nextDocID;
    }
}
