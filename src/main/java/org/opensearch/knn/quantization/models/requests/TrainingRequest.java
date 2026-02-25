/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.requests;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.io.IOException;

/**
 * TrainingRequest represents a request for training a quantizer.
 *
 * @param <T> the type of vectors to be trained.
 */
@Getter
@AllArgsConstructor
public abstract class TrainingRequest<T> {
    /**
     * The total number of vectors in one segment.
     */
    private final int totalNumberOfVectors;
    private final boolean isEnableRandomRotation;

    /**
     * Constructor with only totalNumberOfVectors parameter.
     * Sets isEnableRandomRotation to false by default.
     *
     * @param totalNumberOfVectors the total number of vectors in one segment
     */
    public TrainingRequest(int totalNumberOfVectors) {
        this(totalNumberOfVectors, QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION);
    }

    /**
     * Returns the vector corresponding to the specified document ID.
     *
     * @param position the document position.
     * @return the vector corresponding to the specified document ID.
     */
    public abstract T getVectorAtThePosition(int position) throws IOException;

    /**
     * Resets the KNNVectorValues in implementation to enable a fresh iteration by calling the supplier again.
     * If the implementation does not use KNNVectorValues and getVectorAtThePosition is stateless then
     * resetVectorValues is a no-op.
     */
    public abstract void resetVectorValues();
}
