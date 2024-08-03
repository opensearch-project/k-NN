/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import lombok.experimental.UtilityClass;

import java.util.BitSet;

/**
 * Utility class providing common methods for quantizer operations, such as parameter validation and
 * extraction. This class is designed to be used with various quantizer implementations that require
 * consistent handling of training requests and sampled indices.
 */
@UtilityClass
class QuantizerHelper {
    /**
     * Calculates the mean vector from a set of sampled vectors.
     *
     * <p>This method takes a {@link TrainingRequest} object and an array of sampled indices,
     * retrieves the vectors corresponding to these indices, and calculates the mean vector.
     * Each element of the mean vector is computed as the average of the corresponding elements
     * of the sampled vectors.</p>
     *
     * @param samplingRequest The {@link TrainingRequest} containing the dataset and methods to access vectors by their indices.
     * @param sampledIndices An array of indices representing the sampled vectors to be used for mean calculation.
     * @return A float array representing the mean vector of the sampled vectors.
     * @throws IllegalArgumentException If any of the vectors at the sampled indices are null.
     * @throws IllegalStateException If the mean array is unexpectedly null after processing the vectors.
     */
    static float[] calculateMeanThresholds(TrainingRequest<float[]> samplingRequest, BitSet sampledIndices) {
        int totalSamples = sampledIndices.cardinality();
        float[] mean = null;
        for (int docId = sampledIndices.nextSetBit(0); docId >= 0; docId = sampledIndices.nextSetBit(docId + 1)) {
            float[] vector = samplingRequest.getVectorByDocId(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }
            if (mean == null) {
                mean = new float[vector.length];
            }
            for (int j = 0; j < vector.length; j++) {
                mean[j] += vector[j];
            }
        }
        if (mean == null) {
            throw new IllegalStateException("Mean array should not be null after processing vectors.");
        }
        for (int j = 0; j < mean.length; j++) {
            mean[j] /= totalSamples;
        }
        return mean;
    }

    /**
     * Calculates the sum, sum of squares, mean, and standard deviation for each dimension in a single pass.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @param sampledIndices  the indices of the sampled vectors.
     * @param meanArray      the array to store the sum and then the mean of each dimension.
     * @param stdDevArray  the array to store the sum of squares and then the standard deviation of each dimension.
     */
    static void calculateSumMeanAndStdDev(
        TrainingRequest<float[]> trainingRequest,
        BitSet sampledIndices,
        float[] meanArray,
        float[] stdDevArray
    ) {
        int totalSamples = sampledIndices.cardinality();
        int dimension = meanArray.length;

        // Single pass to calculate sum and sum of squares
        for (int docId = sampledIndices.nextSetBit(0); docId >= 0; docId = sampledIndices.nextSetBit(docId + 1)) {
            float[] vector = trainingRequest.getVectorByDocId(docId);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + docId + " is null.");
            }
            for (int j = 0; j < dimension; j++) {
                meanArray[j] += vector[j];
                stdDevArray[j] += vector[j] * vector[j];
            }
        }

        // Calculate mean and standard deviation in one pass
        for (int j = 0; j < dimension; j++) {
            meanArray[j] = meanArray[j] / totalSamples;
            stdDevArray[j] = (float) Math.sqrt((stdDevArray[j] / totalSamples) - (meanArray[j] * meanArray[j]));
        }
    }
}
