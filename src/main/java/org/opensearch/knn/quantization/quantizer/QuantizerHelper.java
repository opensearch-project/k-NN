/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import lombok.experimental.UtilityClass;

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
     * @param samplingRequest The {@link TrainingRequest} containing the dataset and methods to access vectors by their indices.
     * @param sampledIndices An array of indices representing the sampled vectors to be used for mean calculation.
     * @return A float array representing the mean vector of the sampled vectors.
     * @throws IllegalArgumentException If any of the vectors at the sampled indices are null.
     * @throws IllegalStateException If the mean array is unexpectedly null after processing the vectors.
     */
    static float[] calculateMeanThresholds(TrainingRequest<float[]> samplingRequest, int[] sampledIndices) {
        int totalSamples = sampledIndices.length;
        float[] mean = null;
        for (int docId : sampledIndices) {
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
     * Calculates the mean and StdDev per dimension for sampled vectors.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @param sampledIndices  the indices of the sampled vectors.
     * @param meanArray      the array to store the sum and then the mean of each dimension.
     * @param stdDevArray  the array to store the sum of squares and then the standard deviation of each dimension.
     */
    static void calculateMeanAndStdDev(
        TrainingRequest<float[]> trainingRequest,
        int[] sampledIndices,
        float[] meanArray,
        float[] stdDevArray
    ) {
        int totalSamples = sampledIndices.length;
        int dimension = meanArray.length;
        for (int docId : sampledIndices) {
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
