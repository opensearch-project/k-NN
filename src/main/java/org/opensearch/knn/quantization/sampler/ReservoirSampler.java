/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import lombok.NoArgsConstructor;

import java.util.BitSet;
import java.util.concurrent.ThreadLocalRandom;

/**
 * ReservoirSampler implements the Sampler interface and provides a method for sampling
 * a specified number of indices from a total number of vectors using the reservoir sampling algorithm.
 * This algorithm is particularly useful for randomly sampling a subset of data from a larger set
 * when the total size of the dataset is unknown or very large.
 */
@NoArgsConstructor
final class ReservoirSampler implements Sampler {
    /**
     * Singleton instance holder.
     */
    private static ReservoirSampler instance;

    /**
     * Provides the singleton instance of ReservoirSampler.
     *
     * @return the singleton instance of ReservoirSampler.
     */
    public static synchronized ReservoirSampler getInstance() {
        if (instance == null) {
            instance = new ReservoirSampler();
        }
        return instance;
    }

    /**
     * Samples indices from the range [0, totalNumberOfVectors).
     * If the total number of vectors is less than or equal to the sample size, it returns all indices.
     * Otherwise, it uses the reservoir sampling algorithm to select a random subset.
     *
     * @param totalNumberOfVectors the total number of vectors to sample from.
     * @param sampleSize           the number of indices to sample.
     * @return an array of sampled indices.
     */
    @Override
    public BitSet sample(final int totalNumberOfVectors, final int sampleSize) {
        if (totalNumberOfVectors <= sampleSize) {
            BitSet bitSet = new BitSet(totalNumberOfVectors);
            bitSet.set(0, totalNumberOfVectors);
            return bitSet;
        }
        return reservoirSampleIndices(totalNumberOfVectors, sampleSize);
    }

    /**
     * Applies the reservoir sampling algorithm to select a random sample of indices.
     * This method ensures that each index in the range [0, numVectors) has an equal probability
     * of being included in the sample.
     *
     * Reservoir sampling is particularly useful for selecting a random sample from a large or unknown-sized dataset.
     * For more information on the algorithm, see the following link:
     * <a href="https://en.wikipedia.org/wiki/Reservoir_sampling">Reservoir Sampling - Wikipedia</a>
     *
     * @param numVectors the total number of vectors.
     * @param sampleSize the number of indices to sample.
     * @return a BitSet representing the sampled indices.
     */
    private BitSet reservoirSampleIndices(final int numVectors, final int sampleSize) {
        int[] indices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            indices[i] = i;
        }
        for (int i = sampleSize; i < numVectors; i++) {
            int j = ThreadLocalRandom.current().nextInt(i + 1);
            if (j < sampleSize) {
                indices[j] = i;
            }
        }
        // Using BitSet to track the presence of indices
        BitSet bitSet = new BitSet(numVectors);
        for (int i = 0; i < sampleSize; i++) {
            bitSet.set(indices[i]);
        }
        return bitSet;
    }
}
