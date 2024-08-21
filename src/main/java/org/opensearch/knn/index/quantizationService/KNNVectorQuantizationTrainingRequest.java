/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.quantizationService;

import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * KNNVectorQuantizationTrainingRequest is a concrete implementation of the abstract TrainingRequest class.
 * It provides a mechanism to retrieve float vectors from the KNNVectorValues by document ID.
 */
class KNNVectorQuantizationTrainingRequest<T> extends TrainingRequest<T> {

    private final KNNVectorValues<T> knnVectorValues;
    private int lastIndex;

    /**
     * Constructs a new QuantizationFloatVectorTrainingRequest.
     *
     * @param knnVectorValues the KNNVectorValues instance containing the vectors.
     */
    KNNVectorQuantizationTrainingRequest(KNNVectorValues<T> knnVectorValues) {
        super((int) knnVectorValues.totalLiveDocs());
        this.knnVectorValues = knnVectorValues;
        this.lastIndex = 0;
    }

    /**
     * Retrieves the float vector associated with the specified document ID.
     *
     * @param docId the document ID.
     * @return the float vector corresponding to the specified document ID, or null if the docId is invalid.
     */
    @Override
    public T getVectorByDocId(int docId) {
        try {
            int index = lastIndex;
            while (index <= docId) {
                knnVectorValues.nextDoc();
                index++;
            }
            if (knnVectorValues.docId() == NO_MORE_DOCS) {
                return null;
            }
            lastIndex = index;
            // Return the vector and the updated index
            return knnVectorValues.getVector();
        } catch (Exception e) {
            throw new RuntimeException("Failed to retrieve vector for docId: " + docId, e);
        }
    }
}
