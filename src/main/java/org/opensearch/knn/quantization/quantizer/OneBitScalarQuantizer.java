/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.ADC_CORRECTION_FACTOR;

/**
 * OneBitScalarQuantizer is responsible for quantizing vectors using a single bit per dimension.
 * It computes the mean of each dimension during training and then uses these means as thresholds
 * for quantizing the vectors.
 */
public class OneBitScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int samplingSize; // Sampling size for training
    private final boolean shouldUseRandomRotation;
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
        this(DEFAULT_SAMPLE_SIZE, QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION, SamplingFactory.getSampler(SamplerType.RESERVOIR));
    }

    /**
     * Constructs a OneBitScalarQuantizer with a specified sampling size.
     *
     * @param samplingSize the number of samples to use for training.
     */
    public OneBitScalarQuantizer(final int samplingSize, final boolean shouldUseRandomRotation, final Sampler sampler) {
        this.samplingSize = samplingSize;
        this.shouldUseRandomRotation = shouldUseRandomRotation;
        this.sampler = sampler;
    }

    public OneBitScalarQuantizer(final boolean shouldUseRandomRotation) {
        this(DEFAULT_SAMPLE_SIZE, shouldUseRandomRotation, SamplingFactory.getSampler(SamplerType.RESERVOIR));
    }

    public OneBitScalarQuantizer(final int samplingSize, final Sampler sampler) {
        this.samplingSize = samplingSize;
        this.shouldUseRandomRotation = QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION;
        this.sampler = sampler;
    }

    /**
     * Trains the quantizer by calculating the mean of each dimension from the sampled vectors.
     * These means are used as thresholds in the quantization process.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @return a OneBitScalarQuantizationState containing the calculated means.
     */
    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) throws IOException {
        int[] sampledDocIds = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        return QuantizerHelper.calculateQuantizationState(
            trainingRequest,
            sampledDocIds,
            ScalarQuantizationParams.builder()
                .sqType(ScalarQuantizationType.ONE_BIT)
                .enableRandomRotation(this.shouldUseRandomRotation)
                .build()
        );
    }

    /**
     * Quantizes the provided vector using the given quantization state.
     * It compares each dimension of the vector against the corresponding mean (threshold) to determine the quantized value.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing the means for each dimension.
     * @param output the QuantizationOutput object to store the quantized representation of the vector.
     */
    @Override
    public void quantize(float[] vector, final QuantizationState state, final QuantizationOutput<byte[]> output) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        int vectorLength = vector.length;
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;
        float[] thresholds = binaryState.getMeanThresholds();
        if (thresholds == null || thresholds.length != vectorLength) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }
        float[][] rotationMatrix = binaryState.getRotationMatrix();
        if (rotationMatrix != null) {
            vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        }
        output.prepareQuantizedVector(vectorLength);
        BitPacker.quantizeAndPackBits(vector, thresholds, output.getQuantizedVector());
    }

    /**
     * Transform vector with ADC. ADC allows us to score full-precision query vectors against binary document vectors.
     * The transformation formula is:
     * q_d = (q_d - x_d) / (y_d - x_d) where x_d is the mean of all document entries quantized to 0 (the below threshold mean)
     * and y_d is the mean of all document entries quantized to 1 (the above threshold mean).
     * @param vector array of floats, modified in-place.
     * @param state The {@link QuantizationState} containing the state of the trained quantizer.
     * @param spaceType spaceType (l2 or innerproduct). Used to identify whether an additional correction term should be applied.
     */
    @Override
    public void transformWithADC(float[] vector, final QuantizationState state, final SpaceType spaceType) {
        validateState(state);
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;

        if (shouldDoADCCorrection(spaceType)) {
            transformVectorWithADCCorrection(vector, binaryState);
        } else {
            transformVectorWithADCNoCorrection(vector, binaryState);
        }
    }

    private boolean shouldDoADCCorrection(SpaceType spaceType) {
        // Note that correction will not work for cosine similarity since these vectors are normalized and correction will break
        // normalization.
        // A normalization-aware correction term may be added in the future so we can support inner product spaces.
        return SpaceType.L2.equals(spaceType);
    }

    private void transformVectorWithADCNoCorrection(float[] vector, final OneBitScalarQuantizationState binaryState) {
        for (int i = 0; i < vector.length; ++i) {
            float aboveThreshold = binaryState.getAboveThresholdMeans()[i];
            float belowThreshold = binaryState.getBelowThresholdMeans()[i];

            vector[i] = (vector[i] - belowThreshold) / (aboveThreshold - belowThreshold);
        }
    }

    private void transformVectorWithADCCorrection(float[] vector, final OneBitScalarQuantizationState binaryState) {
        for (int i = 0; i < vector.length; i++) {
            float aboveThreshold = binaryState.getAboveThresholdMeans()[i];
            float belowThreshold = binaryState.getBelowThresholdMeans()[i];
            double correction = Math.pow(aboveThreshold - belowThreshold, ADC_CORRECTION_FACTOR);
            vector[i] = (vector[i] - belowThreshold) / (aboveThreshold - belowThreshold);
            vector[i] = (float) correction * (vector[i] - 0.5f) + 0.5f;
        }
    }

    @Override
    public void transform(float[] vector, final QuantizationState state) {
        if (vector == null) {
            return;
        }
        validateState(state);
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;
        float[][] rotationMatrix = binaryState.getRotationMatrix();
        if (rotationMatrix != null) {
            RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        }

        for (int i = 0; i < vector.length; i++) {
            float aboveThreshold = binaryState.getAboveThresholdMeans()[i];
            float belowThreshold = binaryState.getBelowThresholdMeans()[i];
            vector[i] = (vector[i] - belowThreshold) / (aboveThreshold - belowThreshold);
        }
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
