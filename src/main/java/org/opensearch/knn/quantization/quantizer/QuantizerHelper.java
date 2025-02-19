/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import lombok.experimental.UtilityClass;
import oshi.util.tuples.Pair;

import java.io.IOException;

/**
 * Utility class providing common methods for quantizer operations, such as parameter validation and
 * extraction. This class is designed to be used with various quantizer implementations that require
 * consistent handling of training requests and sampled indices.
 */
@UtilityClass
class QuantizerHelper {

    // This value can change based on Experiments.
    private static final double ROTATION_MATRIX_THRESHOLD = 0.6;

    /**
     * Calculates the quantization state using the provided training data and sampled indices.
     * <p>
     * This method combines the calculation of mean thresholds, average L2/L1 ratio, and
     * below/above threshold means to construct a {@link OneBitScalarQuantizationState}.
     * </p>
     *
     * @param trainingRequest   The {@link TrainingRequest} containing the dataset and access methods for vector retrieval.
     * @param sampledIndices    An array of indices representing the sampled vectors.
     * @param quantizationParams The scalar quantization parameters.
     * @return A fully constructed {@link OneBitScalarQuantizationState}.
     * @throws IOException If an I/O error occurs while retrieving vector data.
     */
    static OneBitScalarQuantizationState calculateQuantizationState(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        ScalarQuantizationParams quantizationParams
    ) throws IOException {
        if (sampledIndices.length == 0) {
            throw new IllegalArgumentException("No samples provided.");
        }

        // Calculate mean thresholds and L2/L1 ratio in a single pass
        Pair<float[], Double> meanAndL2L1 = calculateMeanAndL2L1Ratio(trainingRequest, sampledIndices);
        float[] meanThresholds = meanAndL2L1.getA();
        double averageL2L1Ratio = meanAndL2L1.getB();
        // Apply random rotation if L2/L1 ratio is greater than 0.6
        float[][] rotationMatrix = null;
        if (averageL2L1Ratio > ROTATION_MATRIX_THRESHOLD) {
            int dimensions = meanThresholds.length;
            rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimensions);

            // Apply rotation to mean thresholds
            meanThresholds = RandomGaussianRotation.applyRotation(meanThresholds, rotationMatrix);
        }

        // Calculate below and above threshold means
        trainingRequest.resetVectorValues();
        Pair<float[], float[]> belowAboveMeans = calculateBelowAboveThresholdMeans(trainingRequest, meanThresholds, sampledIndices);
        float[] belowThresholdMeans = belowAboveMeans.getA();
        float[] aboveThresholdMeans = belowAboveMeans.getB();

        // Apply the same rotation to below and above threshold means if rotation was applied
        if (rotationMatrix != null) {
            belowThresholdMeans = RandomGaussianRotation.applyRotation(belowThresholdMeans, rotationMatrix);
            aboveThresholdMeans = RandomGaussianRotation.applyRotation(aboveThresholdMeans, rotationMatrix);
        }

        // Construct and return the quantization state
        return OneBitScalarQuantizationState.builder()
            .quantizationParams(quantizationParams)
            .meanThresholds(meanThresholds)
            .belowThresholdMeans(belowThresholdMeans)
            .aboveThresholdMeans(aboveThresholdMeans)
            .averageL2L1Ratio(averageL2L1Ratio)
            .rotationMatrix(rotationMatrix)
            .build();
    }

    /**
     * Calculates the mean thresholds and average L2/L1 ratio for the given sampled vectors.
     * <p>
     * The mean thresholds are computed by averaging the values across all sampled vectors for each dimension.
     * The average L2/L1 ratio is calculated as the mean of the L2/L1 ratio for each vector.
     * </p>
     *
     * @param trainingRequest The {@link TrainingRequest} containing the dataset and access methods for vector retrieval.
     * @param sampledIndices  An array of indices representing the sampled vectors.
     * @return A {@link Pair} where the first element is the array of mean thresholds (float[]) and the second element
     * is the average L2/L1 ratio (Double).
     * @throws IOException              If an I/O error occurs while retrieving vector data.
     * @throws IllegalArgumentException If any vector at the sampled indices is null.
     */
    private static Pair<float[], Double> calculateMeanAndL2L1Ratio(TrainingRequest<float[]> trainingRequest, int[] sampledIndices)
        throws IOException {
        float[] meanThresholds = null;
        double totalL2L1Ratio = 0.0;
        int totalSamples = sampledIndices.length;

        for (int docId : sampledIndices) {
            float[] vector = trainingRequest.getVectorAtThePosition(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }

            if (meanThresholds == null) {
                meanThresholds = new float[vector.length];
            }

            double l2Norm = 0.0;
            double l1Norm = 0.0;

            for (int j = 0; j < vector.length; j++) {
                float value = vector[j];

                // Accumulate mean
                meanThresholds[j] += value;

                // Accumulate norms
                l2Norm += value * value;
                l1Norm += Math.abs(value);
            }

            // Update L2/L1 ratio for the vector
            totalL2L1Ratio += Math.sqrt(l2Norm) / l1Norm;
        }

        // Finalize mean thresholds
        for (int j = 0; j < meanThresholds.length; j++) {
            meanThresholds[j] /= totalSamples;
        }

        // Calculate average L2/L1 ratio
        double averageL2L1Ratio = totalL2L1Ratio / totalSamples;

        return new Pair<>(meanThresholds, averageL2L1Ratio);
    }

    /**
     * Calculates the below and above threshold means for the given sampled vectors.
     * <p>
     * For each dimension, values are classified as either below or above the mean threshold,
     * and their respective means are calculated.
     * </p>
     *
     * @param trainingRequest The {@link TrainingRequest} containing the dataset and access methods for vector retrieval.
     * @param thresholds      The mean thresholds for each dimension.
     * @param sampledIndices  An array of indices representing the sampled vectors.
     * @return A {@link Pair} containing two float arrays:
     * - The first array represents the below threshold means.
     * - The second array represents the above threshold means.
     * @throws IOException If an I/O error occurs while retrieving vector data.
     */
    private static Pair<float[], float[]> calculateBelowAboveThresholdMeans(
        TrainingRequest<float[]> trainingRequest,
        float[] thresholds,
        int[] sampledIndices
    ) throws IOException {
        int dimension = thresholds.length;
        float[] belowThresholdMeans = new float[dimension];
        float[] aboveThresholdMeans = new float[dimension];
        int[] belowThresholdCounts = new int[dimension];
        int[] aboveThresholdCounts = new int[dimension];

        for (int docId : sampledIndices) {
            float[] vector = trainingRequest.getVectorAtThePosition(docId);
            if (vector == null) {
                continue;
            }

            for (int j = 0; j < dimension; j++) {
                float value = vector[j];

                if (value <= thresholds[j]) {
                    belowThresholdMeans[j] += value;
                    belowThresholdCounts[j]++;
                } else {
                    aboveThresholdMeans[j] += value;
                    aboveThresholdCounts[j]++;
                }
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

        return new Pair<>(belowThresholdMeans, aboveThresholdMeans);
    }

    /**
     * Calculates the mean and standard deviation for each dimension of the vectors in the training request.
     * <p>
     * This method processes the vectors specified by the sampled indices and calculates both the mean and
     * standard deviation in one pass. The results are returned as a pair of arrays: one for the means and
     * one for the standard deviations.
     *
     * @param trainingRequest The request containing the data and parameters for training.
     * @param sampledIndices  An array of document IDs representing the sampled indices to be processed.
     * @return A Pair containing two float arrays: the first array represents the mean of each dimension,
     * and the second array represents the standard deviation of each dimension.
     * @throws IllegalArgumentException if any of the vectors at the sampled indices are null.
     * @throws IllegalStateException    if the mean or standard deviation arrays are not initialized after processing.
     */
    static Pair<float[], float[]> calculateMeanAndStdDev(TrainingRequest<float[]> trainingRequest, int[] sampledIndices)
        throws IOException {
        float[] meanArray = null;
        float[] stdDevArray = null;
        int totalSamples = sampledIndices.length;
        for (int docId : sampledIndices) {
            float[] vector = trainingRequest.getVectorAtThePosition(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }
            int dimension = vector.length;

            // Initialize meanArray and stdDevArray on the first iteration
            if (meanArray == null) {
                meanArray = new float[dimension];
            }
            if (stdDevArray == null) {
                stdDevArray = new float[dimension];
            }

            for (int j = 0; j < dimension; j++) {
                meanArray[j] += vector[j];
                stdDevArray[j] += vector[j] * vector[j];
            }
        }
        if (meanArray == null || stdDevArray == null) {
            throw new IllegalStateException("Mean and StdDev should not be null after processing vectors.");
        }

        // Calculate mean and standard deviation in one pass
        for (int j = 0; j < meanArray.length; j++) {
            meanArray[j] = meanArray[j] / totalSamples;
            stdDevArray[j] = (float) Math.sqrt((stdDevArray[j] / totalSamples) - (meanArray[j] * meanArray[j]));
        }

        // Return both arrays as a Pair
        return new Pair<>(meanArray, stdDevArray);
    }
}
