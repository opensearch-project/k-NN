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

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
public interface Quantizer<T, R> {
    int getSamplingSize();

    default QuantizationState train(TrainingRequest<T> trainingRequest) {
        throw new UnsupportedOperationException("Train method is not supported by this quantizer.");
    }

    default QuantizationOutput<R> quantize(T vector, QuantizationState state) {
        throw new UnsupportedOperationException("Quantize method with state is not supported by this quantizer.");
    }

    default QuantizationOutput<R> quantize(T vector) {
        throw new UnsupportedOperationException("Quantize method without state is not supported by this quantizer.");
    }
}
