/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import oshi.util.tuples.Pair;

import java.io.IOException;

/**
 * MultiBitScalarQuantizer is responsible for quantizing vectors into multi-bit representations per dimension.
 * Unlike the OneBitScalarQuantizer, which uses a single bit per dimension to represent whether a value is above
 * or below a mean threshold, the MultiBitScalarQuantizer allows for multiple bits per dimension, enabling more
 * granular and precise quantization.
 *
 * <p>
 * In a OneBitScalarQuantizer, each dimension of a vector is compared to a single threshold (the mean), and a single
 * bit is used to indicate whether the value is above or below that threshold. This results in a very coarse
 * representation where each dimension is either "on" or "off."
 * </p>
 *
 * <p>
 * The MultiBitScalarQuantizer, on the other hand, uses multiple thresholds per dimension. For example, in a 2-bit
 * quantization scheme, three thresholds are used to divide each dimension into four possible regions. Each region
 * is represented by a unique 2-bit value. This allows for a much finer representation of the data, capturing more
 * nuances in the variation of each dimension.
 * </p>
 *
 * <p>
 * The thresholds in MultiBitScalarQuantizer are calculated based on the mean and standard deviation of the sampled
 * vectors for each dimension. Here's how it works:
 * </p>
 *
 * <ul>
 *     <li>First, the mean and standard deviation are computed for each dimension across the sampled vectors.</li>
 *     <li>For each bit used in the quantization (e.g., 2 bits per coordinate), the thresholds are calculated
 *         using a linear combination of the mean and the standard deviation. The combination coefficients are
 *         determined by the number of bits, allowing the thresholds to split the data into equal probability regions.
 *     </li>
 *     <li>For example, in a 2-bit quantization (which divides data into four regions), the thresholds might be
 *         set at points corresponding to -1 standard deviation, 0 standard deviations (mean), and +1 standard deviation.
 *         This ensures that the data is evenly split into four regions, each represented by a 2-bit value.
 *     </li>
 * </ul>
 *
 * <p>
 * The number of bits per coordinate is determined by the type of scalar quantization being applied, such as 2-bit
 * or 4-bit quantization. The increased number of bits per coordinate in MultiBitScalarQuantizer allows for better
 * preservation of information during the quantization process, making it more suitable for tasks where precision
 * is crucial. However, this comes at the cost of increased storage and computational complexity compared to the
 * simpler OneBitScalarQuantizer.
 * </p>
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
        this(bitsPerCoordinate, DEFAULT_SAMPLE_SIZE, SamplingFactory.getSampler(SamplerType.RESERVOIR));
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
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) throws IOException {
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);

        ScalarQuantizationParams params = (bitsPerCoordinate == 2)
                ? new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT)
                : new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);

        return QuantizerHelper.calculateQuantizationState(
                trainingRequest, sampledIndices, params, bitsPerCoordinate
        );
    }

    /**
     * Quantizes the provided vector using the provided quantization state, producing a quantized output.
     * The vector is quantized based on the thresholds in the quantization state.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing threshold information.
     * @param output the QuantizationOutput object to store the quantized representation of the vector.
     */
    @Override
    public void quantize(float[] vector, final QuantizationState state, final QuantizationOutput<byte[]> output) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        int vectorLength = vector.length;
        MultiBitScalarQuantizationState multiBitState = (MultiBitScalarQuantizationState) state;
        float[][] thresholds = multiBitState.getThresholds();
        if (thresholds == null || thresholds[0].length != vector.length) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }
        float[][] rotationMatrix = multiBitState.getRotationMatrix();
        if (rotationMatrix != null) {
            vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        }
        output.prepareQuantizedVector(vectorLength);
        BitPacker.quantizeAndPackBits(vector, thresholds, bitsPerCoordinate, output.getQuantizedVector());
    }

    /**
     * Calculates the thresholds for quantization based on mean and standard deviation.
     *
     * @param meanArray      the mean for each dimension.
     * @param stdDevArray    the standard deviation for each dimension.
     * @return the thresholds for quantization.
     */
    private float[][] calculateThresholds(final float[] meanArray, final float[] stdDevArray) {
        int dimension = meanArray.length;
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

    /**
     * Returns the number of bits per coordinate used by this quantizer.
     *
     * @return the number of bits per coordinate.
     */
    public int getBitsPerCoordinate() {
        return bitsPerCoordinate;
    }
}
