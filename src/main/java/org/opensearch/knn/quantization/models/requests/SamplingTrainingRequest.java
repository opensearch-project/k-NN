/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.models.requests;

import org.opensearch.knn.quantization.sampler.Sampler;

import java.util.List;

public class SamplingTrainingRequest<T> extends TrainingRequest<T> {
    private TrainingRequest<T> originalRequest;
    private int[] sampledIndices;

    public SamplingTrainingRequest(TrainingRequest<T> originalRequest, Sampler sampler, int sampleSize) {
        super(originalRequest.getParams(), originalRequest.getTotalNumberOfVectors());
        this.originalRequest = originalRequest;
        this.sampledIndices = sampler.sample(originalRequest.getTotalNumberOfVectors(), sampleSize);
    }

    @Override
    public T getVector() {
        return originalRequest.getVector();
    }

    @Override
    public T getVectorByDocId(int docId) {
        return originalRequest.getVectorByDocId(docId);
    }

    public int[] getSampledIndices() {
        return sampledIndices;
    }
}
