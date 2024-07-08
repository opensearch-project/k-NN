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

package org.opensearch.knn.quantization;

import org.opensearch.knn.quantization.factory.QuantizerFactory;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.SamplingTrainingRequest;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.Quantizer;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

public class QuantizationManager {
    private static QuantizationManager instance;

    private QuantizationManager() {}

    public static QuantizationManager getInstance() {
        if (instance == null) {
            instance = new QuantizationManager();
        }
        return instance;
    }
    public <T, R> QuantizationState train(TrainingRequest<T> trainingRequest) {
        Quantizer<T, R> quantizer = (Quantizer<T, R>) getQuantizer(trainingRequest.getParams());
        int sampleSize = quantizer.getSamplingSize();
        Sampler sampler = SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR);
        TrainingRequest<T> sampledRequest = new SamplingTrainingRequest<>(trainingRequest, sampler, sampleSize);
        return quantizer.train(sampledRequest);
    }
    public Quantizer<?, ?> getQuantizer(QuantizationParams params) {
        return QuantizerFactory.getQuantizer(params);
    }
}

