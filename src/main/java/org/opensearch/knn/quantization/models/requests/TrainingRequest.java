/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.requests;

import lombok.AllArgsConstructor;
import lombok.Getter;

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

    /**
     * Returns the vector corresponding to the specified document ID.
     *
     * @param position the document position.
     * @return the vector corresponding to the specified document ID.
     */
    public abstract T getVectorAtThePosition(int position) throws IOException;

    public abstract void resetVectorValues();
}
