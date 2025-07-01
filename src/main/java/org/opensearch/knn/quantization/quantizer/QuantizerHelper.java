/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.Value;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import lombok.experimental.UtilityClass;
import oshi.util.tuples.Pair;

import java.io.IOException;

/**
 * Utility class for calculating quantization state information for both
 * OneBit and MultiBit scalar quantizers. Handles computing thresholds,
 * below/above mean statistics, and rotation matrix application.
 */
@UtilityClass
class QuantizerHelper {
    private static final int ONE_BIT_NUMBER_OF_BITS_PER_COORDINATE = 1;

    /**
     * Calculate quantization state for a {@link OneBitScalarQuantizationState}.
     *
     * @param trainingRequest      The training request containing the vectors.
     * @param sampledIndices       Sampled vector indices.
     * @param quantizationParams   Scalar quantization parameters.
     * @return A fully constructed {@link OneBitScalarQuantizationState}.
     * @throws IOException If vector retrieval fails.
     */
    static OneBitScalarQuantizationState calculateQuantizationState(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        ScalarQuantizationParams quantizationParams
    ) throws IOException {
        QuantizerHelperResult quantizerHelperResult = calculateQuantizationStateHelper(
            trainingRequest,
            sampledIndices,
            ONE_BIT_NUMBER_OF_BITS_PER_COORDINATE
        );

        return OneBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .meanThresholds(quantizerHelperResult.getThresholds()[0])
            .rotationMatrix(quantizerHelperResult.getRotationMatrix())
            .belowThresholdMeans(quantizerHelperResult.getBelow())
            .aboveThresholdMeans(quantizerHelperResult.getAbove())
            .build();
    }

    /**
     * Calculate quantization state for a {@link MultiBitScalarQuantizationState}.
     *
     * @param trainingRequest      The training request containing vectors.
     * @param sampledIndices       Sampled vector indices.
     * @param quantizationParams   Scalar quantization parameters.
     * @param bitsPerCoordinate    Number of bits per dimension for quantization.
     * @return A fully constructed {@link MultiBitScalarQuantizationState}.
     * @throws IOException If vector retrieval fails.
     */
    static MultiBitScalarQuantizationState calculateQuantizationState(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        ScalarQuantizationParams quantizationParams,
        int bitsPerCoordinate
    ) throws IOException {
        QuantizerHelperResult quantizerHelperResult = calculateQuantizationStateHelper(trainingRequest, sampledIndices, bitsPerCoordinate);

        return MultiBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .thresholds(quantizerHelperResult.getThresholds())
            .rotationMatrix(quantizerHelperResult.getRotationMatrix())
            .build();
    }

    /**
     * Validates that sampled indices are not null or empty.
     *
     * @param sampledIndices Indices to validate.
     */
    private static void validateSampledIndices(int[] sampledIndices) {
        if (sampledIndices == null || sampledIndices.length == 0) {
            throw new IllegalArgumentException("Sampled indices cannot be null or empty.");
        }
    }

    /**
     * Calculates thresholds used for multi-bit quantization.
     *
     * @param mean              Mean of each dimension.
     * @param stdDev            Standard deviation per dimension.
     * @param bitsPerCoordinate Number of bits per coordinate.
     * @return 2D array of thresholds of shape [bits][dimensions].
     */
    protected static float[][] calculateThresholds(float[] mean, float[] stdDev, int bitsPerCoordinate) {
        int dim = mean.length;
        float[][] thresholds = new float[bitsPerCoordinate][dim];
        float coef = bitsPerCoordinate + 1;

        for (int b = 0; b < bitsPerCoordinate; b++) {
            float iCoef = -1 + 2 * (b + 1) / coef;
            for (int d = 0; d < dim; d++) {
                thresholds[b][d] = mean[d] + iCoef * stdDev[d];
            }
        }
        return thresholds;
    }

    @Value
    @Getter
    @Builder
    public class QuantizerHelperResult {
        @NonNull
        float[][] thresholds; // note: this is a (1 x dimension) 2D array for one bit quantization

        float[][] rotationMatrix;

        // below and above thresholds means are used for transforming vector for ADC in one bit paradigm.
        float[] below;
        float[] above;
    }

    private static QuantizerHelperResult calculateQuantizationStateHelper(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        Integer bitsPerCoordinate  // 1 for one-bit, >1 for multi-bit
    ) throws IOException {
        validateSampledIndices(sampledIndices);
        int dim = trainingRequest.getVectorAtThePosition(sampledIndices[0]).length;

        float[][] rotationMatrix = null;
        if (trainingRequest.isEnableRandomRotation()) {
            rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dim);
        }

        float[][] thresholds;

        // note: the vectors are rotated before the mean and stddev are calculated if random rotation is enabled.
        Pair<float[], float[]> meanStd = calculateMeanAndStdDev(trainingRequest, sampledIndices, rotationMatrix);

        thresholds = calculateThresholds(meanStd.getA(), meanStd.getB(), bitsPerCoordinate);

        // if bitsPerCoordinate = 1, there should only be one threshold (used to mean center coordinates).
        if (bitsPerCoordinate == 1) {
            assert thresholds.length == 1;
            // grab above and below threshold means for ADC
            Pair<float[], float[]> belowAbove = calculateBelowAboveThresholdMeans(
                trainingRequest,
                thresholds[0],
                sampledIndices,
                rotationMatrix
            );
            return QuantizerHelperResult.builder()
                .thresholds(thresholds)
                .rotationMatrix(rotationMatrix)
                .below(belowAbove.getA())
                .above(belowAbove.getB())
                .build();
        }

        return QuantizerHelperResult.builder().thresholds(thresholds).rotationMatrix(rotationMatrix).build();
    }

    public static Pair<float[], float[]> calculateMeanAndStdDev(TrainingRequest<float[]> request, int[] sampledIndices) throws IOException {
        return calculateMeanAndStdDev(request, sampledIndices, null);
    }

    /**
     * Calculates per-dimension mean and standard deviation using Welford's online algorithm.
     *
     * @param request         Training request.
     * @param sampledIndices  Sampled vector indices.
     * @return Pair of (means[], stdDevs[]).
     * @throws IOException if vector access fails.
     */
    public static Pair<float[], float[]> calculateMeanAndStdDev(
        TrainingRequest<float[]> request,
        int[] sampledIndices,
        float[][] rotationMatrix
    ) throws IOException {
        float[] mean = null;
        float[] m2 = null;
        int count = 0;

        request.resetVectorValues();
        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);

            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
            }

            if (mean == null) {
                mean = new float[vector.length];
                m2 = new float[vector.length];
            }

            count++;
            for (int i = 0; i < vector.length; i++) {
                float delta = vector[i] - mean[i];
                mean[i] += delta / count;
                float delta2 = vector[i] - mean[i];
                m2[i] += delta * delta2;
            }
        }

        if (mean == null) {
            throw new IllegalStateException("Mean array should not be null after processing vectors.");
        }

        float[] stdDev = new float[mean.length];
        for (int i = 0; i < stdDev.length; i++) {
            stdDev[i] = (float) Math.sqrt(m2[i] / count);
        }

        return new Pair<>(mean, stdDev);
    }

    protected static Pair<float[], float[]> calculateBelowAboveThresholdMeans(
        TrainingRequest<float[]> request,
        float[] thresholds,
        int[] sampledIndices,
        float[][] rotationMatrix
    ) throws IOException {
        int dim = thresholds.length;
        float[] below = new float[dim], above = new float[dim];
        int[] belowCount = new int[dim], aboveCount = new int[dim];
        request.resetVectorValues();
        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);

            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            // we may also need to rotate the vector here.
            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
            }

            for (int d = 0; d < dim; d++) {
                if (vector[d] <= thresholds[d]) {
                    below[d] += vector[d];
                    belowCount[d]++;
                } else {
                    above[d] += vector[d];
                    aboveCount[d]++;
                }
            }
        }

        for (int d = 0; d < dim; d++) {
            if (belowCount[d] > 0) below[d] /= belowCount[d];
            if (aboveCount[d] > 0) above[d] /= aboveCount[d];
        }

        return new Pair<>(below, above);
    }
}
