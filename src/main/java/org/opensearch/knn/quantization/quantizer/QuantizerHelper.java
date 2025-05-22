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
import lombok.extern.log4j.Log4j2;
import oshi.util.tuples.Pair;

import java.io.IOException;

/**
 * Utility class for calculating quantization state information for both
 * OneBit and MultiBit scalar quantizers. Handles computing thresholds,
 * below/above mean statistics, L2/L1 ratios, and rotation matrix application.
 */
@Log4j2
@UtilityClass
class QuantizerHelper {
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
        int first_vec_idx = sampledIndices[0];
        int dim = trainingRequest.getVectorAtThePosition(first_vec_idx).length;

        trainingRequest.resetVectorValues();
        float[][] rotationMatrix = null;
        if (quantizationParams.isEnableRandomRotation()) {
            rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dim);
        }
        Pair<float[], Double> meanAndL2L1 = calculateMeanAndL2L1Ratio(trainingRequest, sampledIndices, rotationMatrix);

        float[] meanThresholds = meanAndL2L1.getA();
        double averageL2L1Ratio = meanAndL2L1.getB();

        trainingRequest.resetVectorValues();
        Pair<float[], float[]> belowAbove = calculateBelowAboveThresholdMeans(
            trainingRequest,
            meanThresholds,
            sampledIndices,
            rotationMatrix
        );

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
        int first_vec_idx = sampledIndices[0];
        int dim = trainingRequest.getVectorAtThePosition(first_vec_idx).length;

        float[][] maybeRotationMatrix = null;

        if (quantizationParams.isEnableRandomRotation()) {
            maybeRotationMatrix = RandomGaussianRotation.generateRotationMatrix(dim);
        }

        Pair<float[], float[]> meanStd = calculateMeanAndStdDev(trainingRequest, sampledIndices, maybeRotationMatrix);
        trainingRequest.resetVectorValues();
        Pair<float[], Double> meanL2L1 = calculateMeanAndL2L1Ratio(trainingRequest, sampledIndices, maybeRotationMatrix);

        float[][] thresholds = calculateThresholds(meanStd.getA(), meanStd.getB(), bitsPerCoordinate);

        // get above/below thresholds before rotation.
        trainingRequest.resetVectorValues();
        Pair<float[], float[]> belowAbove = calculateBelowAboveThresholdMeans(
            trainingRequest,
            thresholds,
            bitsPerCoordinate,
            sampledIndices,
            maybeRotationMatrix
        );

        return MultiBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .thresholds(thresholds)
            .belowThresholdMeans(belowAbove.getA())
            .aboveThresholdMeans(belowAbove.getB())
            .averageL2L1Ratio(meanL2L1.getB())
            .rotationMatrix(maybeRotationMatrix)
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
            } // b = 9
              // -1 + 2 / 2
              // 0
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
        int[] sampledIndices,
        float[][] rotationMatrix
    ) throws IOException {
        int dim = thresholds.length;
        float[] below = new float[dim], above = new float[dim];
        int[] belowCount = new int[dim], aboveCount = new int[dim];

        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId); // vector is not rotated.

            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }
            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
            }

            for (int d = 0; d < dim; d++) {
                if (vector[d] <= thresholds[d]) { // thresholds have been rotated exactly when vector has been rotated.
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
        int[] sampledIndices,
        float[][] rotationMatrix
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

            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
            }

            for (int d = 0; d < dim; d++) {
                int quantBits = 0;
                for (int b = 0; b < bitsPerCoordinate; b++) {
                    // determine if the vector exceeds thresholds.
                    // if each vector component exceed all the thresholds, quantBits is 1111 (for 4 bit).
                    // if it exceeds only 2 thresholds, quantBits is 1100.
                    // needs to exceed all thresholds to contribute to the above mean.
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

    public static Pair<float[], float[]> calculateMeanAndStdDev(TrainingRequest<float[]> request, int[] sampledIndices) throws IOException {
        return calculateMeanAndStdDev(request, sampledIndices, null);
    }

    /**
     * Calculates per-dimension mean and standard deviation.
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
        float[] mean = null, sumSq = null;

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
                sumSq = new float[vector.length];
            }

            for (int i = 0; i < vector.length; i++) {
                mean[i] += vector[i];
                sumSq[i] += vector[i] * vector[i];
            }
        }

        if (mean == null) {
            throw new IllegalStateException("Mean array should not be null after processing vectors.");
        }

        int n = sampledIndices.length;
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= n;
            // equivalent to standard deviation via algebra
            sumSq[i] = (float) Math.sqrt((sumSq[i] / n) - (mean[i] * mean[i]));
        }

        return new Pair<>(mean, sumSq);
    }

    public static Pair<float[], Double> calculateMeanAndL2L1Ratio(TrainingRequest<float[]> request, int[] sampledIndices)
        throws IOException {
        return calculateMeanAndL2L1Ratio(request, sampledIndices, null);
    }

    /**
     * Calculates per-dimension mean and average L2/L1 norm ratio.
     *
     * @param request         Training request.
     * @param sampledIndices  Sampled vector indices.
     * @return Pair of (means[], average L2/L1 ratio).
     * @throws IOException if vector access fails.
     */
    public static Pair<float[], Double> calculateMeanAndL2L1Ratio(
        TrainingRequest<float[]> request,
        int[] sampledIndices,
        float[][] rotationMatrix
    ) throws IOException {
        float[] mean = null;
        double totalL2L1 = 0.0;
        int n = sampledIndices.length;

        for (int docId : sampledIndices) {
            float[] vector = request.getVectorAtThePosition(docId);

            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            if (rotationMatrix != null) {
                vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
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
