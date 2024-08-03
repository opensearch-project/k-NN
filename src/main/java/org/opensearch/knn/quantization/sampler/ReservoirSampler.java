/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * ReservoirSampler implements the Sampler interface and provides a method for sampling
 * a specified number of indices from a total number of vectors using the reservoir sampling algorithm.
 * This algorithm is particularly useful for randomly sampling a subset of data from a larger set
 * when the total size of the dataset is unknown or very large.
 */
final class ReservoirSampler implements Sampler {

    private final Random random;

    /**
     * Constructs a ReservoirSampler with a new Random instance.
     */
    public ReservoirSampler() {
        this(ThreadLocalRandom.current());
    }

    /**
     * Constructs a ReservoirSampler with a specified random seed for reproducibility.
     *
     * @param seed the seed for the random number generator.
     */
    public ReservoirSampler(final long seed) {
        this(new Random(seed));
    }

    /**
     * Constructs a ReservoirSampler with a specified Random instance.
     *
     * @param random the Random instance for generating random numbers.
     */
    public ReservoirSampler(final Random random) {
        this.random = random;
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
    public int[] sample(final int totalNumberOfVectors, final int sampleSize) {
        if (totalNumberOfVectors <= sampleSize) {
            return IntStream.range(0, totalNumberOfVectors).toArray();
        }
        return reservoirSampleIndices(totalNumberOfVectors, sampleSize);
    }

    /**
     * Applies the reservoir sampling algorithm to select a random sample of indices.
     * This method ensures that each index in the range [0, numVectors) has an equal probability
     * of being included in the sample.
     *
     * @param numVectors the total number of vectors.
     * @param sampleSize the number of indices to sample.
     * @return an array of sampled indices.
     */
    private int[] reservoirSampleIndices(final int numVectors, final int sampleSize) {
        int[] indices = IntStream.range(0, sampleSize).toArray();
        for (int i = sampleSize; i < numVectors; i++) {
            int j = random.nextInt(i + 1);
            if (j < sampleSize) {
                indices[j] = i;
            }
        }
        Arrays.sort(indices);
        return indices;
    }
}
