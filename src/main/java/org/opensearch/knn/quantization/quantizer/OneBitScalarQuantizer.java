/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import org.opensearch.knn.quantization.util.BitPacker;
import org.opensearch.knn.quantization.util.QuantizerHelper;

import java.util.Collections;

/**
 * OneBitScalarQuantizer is responsible for quantizing vectors using a single bit per dimension.
 * It computes the mean of each dimension during training and then uses these means as thresholds
 * for quantizing the vectors.
 */
public class OneBitScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int samplingSize; // Sampling size for training
    private static final boolean IS_TRAINING_REQUIRED = true;
    private final Sampler sampler; // Sampler for training
    // Currently Lucene has sampling size as
    // 25000 for segment level training , Keeping same
    // to having consistent, Will revisit
    // if this requires change
    private static final int DEFAULT_SAMPLE_SIZE = 25000;

    /**
     * Constructs a OneBitScalarQuantizer with a default sampling size of 25000.
     */
    public OneBitScalarQuantizer() {
        this(DEFAULT_SAMPLE_SIZE, SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR));
    }

    /**
     * Constructs a OneBitScalarQuantizer with a specified sampling size.
     *
     * @param samplingSize the number of samples to use for training.
     */
    public OneBitScalarQuantizer(final int samplingSize, final Sampler sampler) {

        this.samplingSize = samplingSize;
        this.sampler = sampler;
        ;
    }

    /**
     * Trains the quantizer by calculating the mean of each dimension from the sampled vectors.
     * These means are used as thresholds in the quantization process.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @return a OneBitScalarQuantizationState containing the calculated means.
     */
    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) {
        SQParams params = QuantizerHelper.validateAndExtractParams(trainingRequest);
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        float[] mean = QuantizerHelper.calculateMean(trainingRequest, sampledIndices);
        return new OneBitScalarQuantizationState(params, mean);
    }

    /**
     * Quantizes the provided vector using the given quantization state.
     * It compares each dimension of the vector against the corresponding mean (threshold) to determine the quantized value.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing the means for each dimension.
     * @return a BinaryQuantizationOutput containing the quantized data.
     */
    @Override
    public QuantizationOutput<byte[]> quantize(final float[] vector, final QuantizationState state) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;
        float[] thresholds = binaryState.getMeanThresholds();
        if (thresholds == null || thresholds.length != vector.length) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }
        byte[] quantizedVector = new byte[vector.length];
        for (int i = 0; i < vector.length; i++) {
            quantizedVector[i] = (byte) (vector[i] > thresholds[i] ? 1 : 0);
        }
        return new BinaryQuantizationOutput(BitPacker.packBits(Collections.singletonList(quantizedVector)));
    }

    /**
     * Validates the quantization state to ensure it is of the expected type.
     *
     * @param state the quantization state to validate.
     * @throws IllegalArgumentException if the state is not an instance of OneBitScalarQuantizationState.
     */
    private void validateState(final QuantizationState state) {
        if (!(state instanceof OneBitScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type OneBitScalarQuantizationState.");
        }
    }
}
