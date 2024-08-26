/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.quantizationService;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * KNNVectorQuantizationTrainingRequest is a concrete implementation of the abstract TrainingRequest class.
 * It provides a mechanism to retrieve float vectors from the KNNVectorValues by document ID.
 */
@Log4j2
final class KNNVectorQuantizationTrainingRequest<T> extends TrainingRequest<T> {

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
     * Retrieves the vector associated with the specified document ID.
     *
     * @param position the document ID.
     * @return the float vector corresponding to the specified document ID, or null if the docId is invalid.
     */
    @Override
    public T getVectorAtThePosition(int position) throws IOException {
        while (lastIndex <= position) {
            lastIndex++;
            if (knnVectorValues.docId() == NO_MORE_DOCS) {
                return null;
            }
            knnVectorValues.nextDoc();
        }
        // Return the vector and the updated index
        return knnVectorValues.getVector();
    }
}
