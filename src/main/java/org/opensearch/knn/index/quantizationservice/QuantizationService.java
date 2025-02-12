/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.quantizationservice;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.factory.QuantizerFactory;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.quantizer.Quantizer;
import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractQuantizationConfig;

/**
 * A singleton class responsible for handling the quantization process, including training a quantizer
 * and applying quantization to vectors. This class is designed to be thread-safe.
 *
 * @param <T> The type of the input vectors to be quantized.
 * @param <R> The type of the quantized output vectors.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class QuantizationService<T, R> {

    /**
     * The singleton instance of the {@link QuantizationService} class.
     */
    private static final QuantizationService<?, ?> INSTANCE = new QuantizationService<>();

    /**
     * Returns the singleton instance of the {@link QuantizationService} class.
     *
     * @param <T> The type of the input vectors to be quantized.
     * @param <R> The type of the quantized output vectors.
     * @return The singleton instance of {@link QuantizationService}.
     */
    public static <T, R> QuantizationService<T, R> getInstance() {
        return (QuantizationService<T, R>) INSTANCE;
    }

    /**
     * Trains a quantizer using the provided {@link KNNVectorValues} and returns the resulting
     * {@link QuantizationState}. The quantizer is determined based on the given {@link QuantizationParams}.
     *
     * @param quantizationParams The {@link QuantizationParams} containing the parameters for quantization.
     * @param knnVectorValuesSupplier The {@link KNNVectorValues} representing the vector data to be used for training.
     * @return The {@link QuantizationState} containing the state of the trained quantizer.
     * @throws IOException If an I/O error occurs during the training process.
     */
    public QuantizationState train(
        final QuantizationParams quantizationParams,
        final Supplier<KNNVectorValues<T>> knnVectorValuesSupplier,
        final long liveDocs
    ) throws IOException {
        Quantizer<T, R> quantizer = QuantizerFactory.getQuantizer(quantizationParams);

        // Create the training request using the supplier
        KNNVectorQuantizationTrainingRequest<T> trainingRequest = new KNNVectorQuantizationTrainingRequest<>(
            knnVectorValuesSupplier,
            liveDocs
        );

        // Train the quantizer and return the quantization state
        return quantizer.train(trainingRequest);
    }

    /**
     * Applies quantization to the given vector using the specified {@link QuantizationState} and
     * {@link QuantizationOutput}.
     *
     * @param quantizationState The {@link QuantizationState} containing the state of the trained quantizer.
     * @param vector The vector to be quantized.
     * @param quantizationOutput The {@link QuantizationOutput} to store the quantized vector.
     * @return The quantized vector as an object of type {@code R}.
     */
    public R quantize(final QuantizationState quantizationState, final T vector, final QuantizationOutput<R> quantizationOutput) {
        Quantizer<T, R> quantizer = QuantizerFactory.getQuantizer(quantizationState.getQuantizationParams());
        quantizer.quantize(vector, quantizationState, quantizationOutput);
        return quantizationOutput.getQuantizedVector();
    }

    /**
     * Retrieves quantization parameters from the FieldInfo.
     */
    public QuantizationParams getQuantizationParams(final FieldInfo fieldInfo) {
        QuantizationConfig quantizationConfig = extractQuantizationConfig(fieldInfo);
        if (quantizationConfig != QuantizationConfig.EMPTY && quantizationConfig.getQuantizationType() != null) {
            return new ScalarQuantizationParams(quantizationConfig.getQuantizationType());
        }
        return null;
    }

    /**
     * Retrieves the appropriate {@link VectorDataType} to be used during the transfer of vectors for indexing or merging.
     * This method is intended to determine the correct vector data type based on the provided {@link FieldInfo}.
     *
     * @param fieldInfo The {@link FieldInfo} object containing metadata about the field for which the vector data type
     *                  is being determined.
     * @return The {@link VectorDataType} to be used during the vector transfer process
     */
    public VectorDataType getVectorDataTypeForTransfer(final FieldInfo fieldInfo) {
        QuantizationConfig quantizationConfig = extractQuantizationConfig(fieldInfo);
        if (quantizationConfig != QuantizationConfig.EMPTY && quantizationConfig.getQuantizationType() != null) {
            return VectorDataType.BINARY;
        }
        return null;
    }

    /**
     * Creates the appropriate {@link QuantizationOutput} based on the given {@link QuantizationParams}.
     *
     * @param quantizationParams The {@link QuantizationParams} containing the parameters for quantization.
     * @return The {@link QuantizationOutput} corresponding to the provided parameters.
     * @throws IllegalArgumentException If the quantization parameters are unsupported.
     */
    @SuppressWarnings("unchecked")
    public QuantizationOutput<R> createQuantizationOutput(final QuantizationParams quantizationParams) {
        if (quantizationParams instanceof ScalarQuantizationParams) {
            ScalarQuantizationParams scalarParams = (ScalarQuantizationParams) quantizationParams;
            return (QuantizationOutput<R>) new BinaryQuantizationOutput(scalarParams.getSqType().getId());
        }
        throw new IllegalArgumentException("Unsupported quantization parameters: " + quantizationParams.getClass().getName());
    }
}
