/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

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
 * below/above mean statistics, L2/L1 ratios, and rotation matrix application.
 */
@UtilityClass
class QuantizerHelper {

    /**
     * Threshold for triggering random rotation during training.
     * If average L2/L1 ratio exceeds this value, rotation will be applied.
     */
    private static final double ROTATION_MATRIX_THRESHOLD = 0.6;

    // ========================= ONE BIT ========================= //

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
        validateSampledIndices(sampledIndices);

        Pair<float[], Double> meanAndL2L1 = calculateMeanAndL2L1Ratio(trainingRequest, sampledIndices);
        float[] meanThresholds = meanAndL2L1.getA();
        double averageL2L1Ratio = meanAndL2L1.getB();

        float[][] rotationMatrix = maybeApplyRotation(meanThresholds, averageL2L1Ratio);
        if (rotationMatrix != null) {
            meanThresholds = RandomGaussianRotation.applyRotation(meanThresholds, rotationMatrix);
        }

        trainingRequest.resetVectorValues();
        Pair<float[], float[]> belowAbove = calculateBelowAboveThresholdMeans(trainingRequest, meanThresholds, sampledIndices);

        if (rotationMatrix != null) {
            belowAbove = new Pair<>(
                    RandomGaussianRotation.applyRotation(belowAbove.getA(), rotationMatrix),
                    RandomGaussianRotation.applyRotation(belowAbove.getB(), rotationMatrix)
            );
        }

        return OneBitScalarQuantizationState.builder()
                .quantizationParams(quantizationParams)
                .meanThresholds(meanThresholds)
                .belowThresholdMeans(belowAbove.getA())
                .aboveThresholdMeans(belowAbove.getB())
                .averageL2L1Ratio(averageL2L1Ratio)
                .rotationMatrix(rotationMatrix)
                .build();
    }

    // ========================= MULTI BIT ========================= //

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
        validateSampledIndices(sampledIndices);

        Pair<float[], float[]> meanStd = calculateMeanAndStdDev(trainingRequest, sampledIndices);
        trainingRequest.resetVectorValues();
        Pair<float[], Double> meanL2L1 = calculateMeanAndL2L1Ratio(trainingRequest, sampledIndices);

        float[][] thresholds = calculateThresholds(meanStd.getA(), meanStd.getB(), bitsPerCoordinate);
        float[][] rotationMatrix = null;

        if (meanL2L1.getB() > ROTATION_MATRIX_THRESHOLD) {
            rotationMatrix = RandomGaussianRotation.generateRotationMatrix(meanStd.getA().length);
            for (int i = 0; i < thresholds.length; i++) {
                thresholds[i] = RandomGaussianRotation.applyRotation(thresholds[i], rotationMatrix);
            }
        }

        trainingRequest.resetVectorValues();
        Pair<float[], float[]> belowAbove = calculateBelowAboveThresholdMeans(trainingRequest, thresholds, bitsPerCoordinate, sampledIndices);

        if (rotationMatrix != null) {
            belowAbove = new Pair<>(
                    RandomGaussianRotation.applyRotation(belowAbove.getA(), rotationMatrix),
                    RandomGaussianRotation.applyRotation(belowAbove.getB(), rotationMatrix)
            );
        }

        return MultiBitScalarQuantizationState.builder()
                .quantizationParams(quantizationParams)
                .thresholds(thresholds)
                .belowThresholdMeans(belowAbove.getA())
                .aboveThresholdMeans(belowAbove.getB())
                .averageL2L1Ratio(meanL2L1.getB())
                .rotationMatrix(rotationMatrix)
                .build();
    }

    // ========================= INTERNAL HELPERS ========================= //

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
     * Conditionally generates a random rotation matrix if L2/L1 ratio is above threshold.
     *
     * @param baseVector Input vector to determine rotation dimensions.
     * @param l2l1Ratio  L2/L1 ratio of the training vectors.
     * @return A 2D float rotation matrix or null.
     */
    private static float[][] maybeApplyRotation(float[] baseVector, double l2l1Ratio) {
        return l2l1Ratio > ROTATION_MATRIX_THRESHOLD
                ? RandomGaussianRotation.generateRotationMatrix(baseVector.length)
                : null;
    }

    /**
     * Calculates thresholds used for multi-bit quantization.
     *
     * @param mean              Mean of each dimension.
     * @param stdDev            Standard deviation per dimension.
     * @param bitsPerCoordinate Number of bits per coordinate.
     * @return 2D array of thresholds of shape [bits][dimensions].
     */
    private static float[][] calculateThresholds(float[] mean, float[] stdDev, int bitsPerCoordinate) {
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

    /**
     * Calculates the below and above threshold means for a one-bit quantizer.
     *
     * @param request         Training data request.
     * @param thresholds      Thresholds per dimension.
     * @param sampledIndices  Indices to sample.
     * @return Pair of (belowMeans, aboveMeans).
     * @throws IOException if vector access fails.
     */
    private static Pair<float[], float[]> calculateBelowAboveThresholdMeans(
            TrainingRequest<float[]> request,
            float[] thresholds,
            int[] sampledIndices
    ) throws IOException {
        int dim = thresholds.length;
        float[] below = new float[dim], above = new float[dim];
        int[] belowCount = new int[dim], aboveCount = new int[dim];

        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
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

    /**
     * Calculates below/above means for a multi-bit quantizer.
     *
     * @param request            Training data request.
     * @param thresholds         Multi-bit thresholds [bits][dims].
     * @param bitsPerCoordinate  Bits per coordinate.
     * @param sampledIndices     Sampled doc IDs.
     * @return Pair of (belowMeans, aboveMeans).
     * @throws IOException if vector access fails.
     */
    private static Pair<float[], float[]> calculateBelowAboveThresholdMeans(
            TrainingRequest<float[]> request,
            float[][] thresholds,
            int bitsPerCoordinate,
            int[] sampledIndices
    ) throws IOException {
        int dim = thresholds[0].length;
        float[] below = new float[dim], above = new float[dim];
        int[] belowCount = new int[dim], aboveCount = new int[dim];
        int fullBits = (1 << bitsPerCoordinate) - 1;

        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }


            for (int d = 0; d < dim; d++) {
                int quantBits = 0;
                for (int b = 0; b < bitsPerCoordinate; b++) {
                    if (vector[d] > thresholds[b][d]) {
                        quantBits |= (1 << (bitsPerCoordinate - b - 1));
                    }
                }

                if (quantBits == 0) {
                    below[d] += vector[d];
                    belowCount[d]++;
                } else if (quantBits == fullBits) {
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

    /**
     * Calculates per-dimension mean and standard deviation.
     *
     * @param request         Training request.
     * @param sampledIndices  Sampled vector indices.
     * @return Pair of (means[], stdDevs[]).
     * @throws IOException if vector access fails.
     */
    public static Pair<float[], float[]> calculateMeanAndStdDev(TrainingRequest<float[]> request, int[] sampledIndices) throws IOException {
        float[] mean = null, sumSq = null;

        for (int id : sampledIndices) {
            float[] vec = request.getVectorAtThePosition(id);
            if (vec == null) continue;

            if (mean == null) {
                mean = new float[vec.length];
                sumSq = new float[vec.length];
            }

            for (int i = 0; i < vec.length; i++) {
                mean[i] += vec[i];
                sumSq[i] += vec[i] * vec[i];
            }
        }

        // Finalize means
        for (int j = 0; j < dimension; j++) {
            if (belowThresholdCounts[j] > 0) {
                belowThresholdMeans[j] /= belowThresholdCounts[j];
            }
            if (aboveThresholdCounts[j] > 0) {
                aboveThresholdMeans[j] /= aboveThresholdCounts[j];
            }
        }

        int n = sampledIndices.length;
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= n;
            sumSq[i] = (float) Math.sqrt((sumSq[i] / n) - (mean[i] * mean[i]));
        }

        return new Pair<>(mean, sumSq);
    }

    /**
     * Calculates per-dimension mean and average L2/L1 norm ratio.
     *
     * @param request         Training request.
     * @param sampledIndices  Sampled vector indices.
     * @return Pair of (means[], average L2/L1 ratio).
     * @throws IOException if vector access fails.
     */
    public static Pair<float[], Double> calculateMeanAndL2L1Ratio(TrainingRequest<float[]> request, int[] sampledIndices) throws IOException {
        float[] mean = null;
        double totalL2L1 = 0.0;
        int n = sampledIndices.length;

        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            if (mean == null) mean = new float[vector.length];

            double l2 = 0, l1 = 0;
            for (int i = 0; i < vector.length; i++) {
                mean[i] += vector[i];
                l2 += vector[i] * vector[i];
                l1 += Math.abs(vector[i]);
            }

            totalL2L1 += Math.sqrt(l2) / l1;
        }

        for (int i = 0; i < mean.length; i++) {
            mean[i] /= n;
        }

        return new Pair<>(mean, totalL2L1 / n);
    }
}
