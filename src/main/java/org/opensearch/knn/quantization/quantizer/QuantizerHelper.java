/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

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
    /**
     * Calculates the mean vector from a set of sampled vectors.
     *
     * @param samplingRequest The {@link TrainingRequest} containing the dataset and methods to access vectors by their indices.
     * @param sampledIndices  An array of indices representing the sampled vectors to be used for mean calculation.
     * @return A float array representing the mean vector of the sampled vectors.
     * @throws IllegalArgumentException If any of the vectors at the sampled indices are null.
     * @throws IllegalStateException    If the mean array is unexpectedly null after processing the vectors.
     */
    static float[] calculateMeanThresholds(TrainingRequest<float[]> samplingRequest, int[] sampledIndices) throws IOException {
        int totalSamples = sampledIndices.length;
        float[] mean = null;
        int lastIndex = 0;
        for (int docId : sampledIndices) {
            float[] vector = samplingRequest.getVectorAtThePosition(docId);
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
        int lastIndex = 0;
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
