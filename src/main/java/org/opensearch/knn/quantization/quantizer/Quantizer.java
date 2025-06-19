/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import java.io.IOException;

import org.opensearch.knn.index.SpaceType;

/**
 * The Quantizer interface defines the methods required for training and quantizing vectors
 * in the context of K-Nearest Neighbors (KNN) and similar machine learning tasks.
 * It supports training to determine quantization parameters and quantizing data vectors
 * based on these parameters.
 *
 * @param <T> The type of the vector or data to be quantized.
 * @param <R> The type of the quantized output, typically a compressed or encoded representation.
 */
public interface Quantizer<T, R> {

    /**
     * Trains the quantizer based on the provided training request. The training process typically
     * involves learning parameters that can be used to quantize vectors.
     *
     * @param trainingRequest the request containing data and parameters for training.
     * @return a QuantizationState containing the learned parameters.
     */
    QuantizationState train(TrainingRequest<T> trainingRequest) throws IOException;

    /**
     * Quantizes the provided vector using the specified quantization state.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing parameters for quantization.
     * @param output the QuantizationOutput object to store the quantized representation of the vector.
     */
    void quantize(T vector, QuantizationState state, QuantizationOutput<R> output);

    /**
     * Transforms the provided vector based on the quantization state. In one bit scalar quantizer this uses ADC.
     * Otherwise it is a noop.
     *
     * @param vector the vector to transform.
     * @param state  the quantization state containing parameters for quantization.
     */
    default void transformWithADC(T vector, QuantizationState state, SpaceType spaceType) {}
}
