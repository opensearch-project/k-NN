/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import org.opensearch.knn.quantization.util.BitPacker;
import org.opensearch.knn.quantization.util.QuantizerHelper;

import java.util.ArrayList;
import java.util.List;

/**
 * MultiBitScalarQuantizer is responsible for quantizing vectors into multi-bit representations per dimension.
 * It supports multiple bits per coordinate, allowing for finer granularity in quantization.
 */
public class MultiBitScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int bitsPerCoordinate; // Number of bits used to quantize each dimension
    private final int samplingSize; // Sampling size for training
    private final Sampler sampler; // Sampler for training
    private static final boolean IS_TRAINING_REQUIRED = true;
    // Currently Lucene has sampling size as
    // 25000 for segment level training , Keeping same
    // to having consistent, Will revisit
    // if this requires change
    private static final int DEFAULT_SAMPLE_SIZE = 25000;

    /**
     * Constructs a MultiBitScalarQuantizer with a specified number of bits per coordinate.
     *
     * @param bitsPerCoordinate the number of bits used per coordinate for quantization.
     */
    public MultiBitScalarQuantizer(final int bitsPerCoordinate) {
        this(bitsPerCoordinate, DEFAULT_SAMPLE_SIZE, SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR));
    }

    /**
     * Constructs a MultiBitScalarQuantizer with a specified number of bits per coordinate and sampling size.
     *
     * @param bitsPerCoordinate the number of bits used per coordinate for quantization.
     * @param samplingSize the number of samples to use for training.
     * @param sampler the sampler to use for training.
     */
    public MultiBitScalarQuantizer(final int bitsPerCoordinate, final int samplingSize, final Sampler sampler) {
        if (bitsPerCoordinate < 2) {
            throw new IllegalArgumentException("bitsPerCoordinate must be greater than or equal to 2 for multibit quantizer.");
        }
        this.bitsPerCoordinate = bitsPerCoordinate;
        this.samplingSize = samplingSize;
        this.sampler = sampler;
    }

    /**
     * Trains the quantizer based on the provided training request, which should be of type SamplingTrainingRequest.
     * The training process calculates the mean and standard deviation for each dimension and then determines
     * threshold values for quantization based on these statistics.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @return a MultiBitScalarQuantizationState containing the computed thresholds.
     */
    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) {
        SQParams params = QuantizerHelper.validateAndExtractParams(trainingRequest);
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);

        int dimension = trainingRequest.getVectorByDocId(sampledIndices[0]).length;
        float[] meanArray = new float[dimension];
        float[] stdDevArray = new float[dimension];
        // Calculate sum, mean, and standard deviation in one pass
        QuantizerHelper.calculateSumMeanAndStdDev(trainingRequest, sampledIndices, meanArray, stdDevArray);
        float[][] thresholds = calculateThresholds(meanArray, stdDevArray, dimension);
        return new MultiBitScalarQuantizationState(params, thresholds);
    }

    /**
     * Quantizes the provided vector using the provided quantization state, producing a quantized output.
     * The vector is quantized based on the thresholds in the quantization state.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing threshold information.
     * @return a BinaryQuantizationOutput containing the quantized data.
     */
    @Override
    public QuantizationOutput<byte[]> quantize(final float[] vector, final QuantizationState state) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        MultiBitScalarQuantizationState multiBitState = (MultiBitScalarQuantizationState) state;
        float[][] thresholds = multiBitState.getThresholds();
        if (thresholds == null || thresholds[0].length != vector.length) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }

        List<byte[]> bitArrays = new ArrayList<>();
        for (int i = 0; i < bitsPerCoordinate; i++) {
            byte[] bitArray = new byte[vector.length];
            for (int j = 0; j < vector.length; j++) {
                bitArray[j] = (byte) (vector[j] > thresholds[i][j] ? 1 : 0);
            }
            bitArrays.add(bitArray);
        }

        return new BinaryQuantizationOutput(BitPacker.packBits(bitArrays));
    }

    /**
     * Calculates the thresholds for quantization based on mean and standard deviation.
     *
     * @param meanArray      the mean for each dimension.
     * @param stdDevArray    the standard deviation for each dimension.
     * @param dimension the number of dimensions in the vectors.
     * @return the thresholds for quantization.
     */
    private float[][] calculateThresholds(final float[] meanArray, final float[] stdDevArray, final int dimension) {
        float[][] thresholds = new float[bitsPerCoordinate][dimension];
        float coef = bitsPerCoordinate + 1;
        for (int i = 0; i < bitsPerCoordinate; i++) {
            float iCoef = -1 + 2 * (i + 1) / coef;
            for (int j = 0; j < dimension; j++) {
                thresholds[i][j] = meanArray[j] + iCoef * stdDevArray[j];
            }
        }
        return thresholds;
    }

    /**
     * Validates the quantization state to ensure it is of the expected type.
     *
     * @param state the quantization state to validate.
     * @throws IllegalArgumentException if the state is not an instance of MultiBitScalarQuantizationState.
     */
    private void validateState(final QuantizationState state) {
        if (!(state instanceof MultiBitScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type MultiBitScalarQuantizationState.");
        }
    }
}
