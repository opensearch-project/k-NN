/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.requests;

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

/**
 * TrainingRequest represents a request for training a quantizer.
 *
 * @param <T> the type of vectors to be trained.
 */
public abstract class TrainingRequest<T> {
    private final QuantizationParams params;
    private final int totalNumberOfVectors;
    private int[] sampledIndices;

    /**
     * Constructs a TrainingRequest with the given parameters and total number of vectors.
     *
     * @param params              the quantization parameters.
     * @param totalNumberOfVectors the total number of vectors.
     */
    protected TrainingRequest(final QuantizationParams params, final int totalNumberOfVectors) {
        this.params = params;
        this.totalNumberOfVectors = totalNumberOfVectors;
    }

    /**
     * Returns the quantization parameters.
     *
     * @return the quantization parameters.
     */
    public QuantizationParams getParams() {
        return params;
    }

    /**
     * Returns the total number of vectors.
     *
     * @return the total number of vectors.
     */
    public int getTotalNumberOfVectors() {
        return totalNumberOfVectors;
    }

    /**
     * Sets the sampled indices for this training request.
     *
     * @param sampledIndices the sampled indices.
     */
    public void setSampledIndices(int[] sampledIndices) {
        this.sampledIndices = sampledIndices;
    }

    /**
     * Returns the sampled indices for this training request.
     *
     * @return the sampled indices.
     */
    public int[] getSampledIndices() {
        return sampledIndices;
    }

    /**
     * Returns the vector corresponding to the specified document ID.
     *
     * @param docId the document ID.
     * @return the vector corresponding to the specified document ID.
     */
    public abstract T getVectorByDocId(int docId);
}
