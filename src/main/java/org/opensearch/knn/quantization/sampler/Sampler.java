/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

/**
 * The Sampler interface defines the contract for sampling strategies
 * used in various quantization processes. Implementations of this
 * interface should provide specific strategies for selecting a sample
 * from a given set of vectors.
 */
public interface Sampler {

    /**
     * Samples a subset of indices from the total number of vectors.
     *
     * @param totalNumberOfVectors the total number of vectors available.
     * @param sampleSize the number of vectors to be sampled.
     * @return an array of integers representing the indices of the sampled vectors.
     * @throws IllegalArgumentException if the sample size is greater than the total number of vectors.
     */
    int[] sample(int totalNumberOfVectors, int sampleSize);
}
