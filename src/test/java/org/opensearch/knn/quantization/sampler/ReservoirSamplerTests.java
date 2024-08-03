/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ReservoirSamplerTests extends KNNTestCase {

    public void testSampleLessThanSampleSize() {
        ReservoirSampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 5;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        int[] expectedIndices = IntStream.range(0, totalNumberOfVectors).toArray();
        assertArrayEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleEqualToSampleSize() {
        ReservoirSampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 10;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        int[] expectedIndices = IntStream.range(0, totalNumberOfVectors).toArray();
        assertArrayEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleGreaterThanSampleSize() {
        ReservoirSampler sampler = new ReservoirSampler(12345); // Fixed seed for reproducibility
        int totalNumberOfVectors = 100;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(sampleSize, sampledIndices.length);
        assertTrue(Arrays.stream(sampledIndices).allMatch(i -> i >= 0 && i < totalNumberOfVectors));
    }

    public void testSampleReproducibility() {
        long seed = 12345L;
        ReservoirSampler sampler1 = new ReservoirSampler(seed);
        ReservoirSampler sampler2 = new ReservoirSampler(seed);
        int totalNumberOfVectors = 100;
        int sampleSize = 10;

        int[] sampledIndices1 = sampler1.sample(totalNumberOfVectors, sampleSize);
        int[] sampledIndices2 = sampler2.sample(totalNumberOfVectors, sampleSize);

        assertArrayEquals(sampledIndices1, sampledIndices2);
    }

    public void testSampleRandomness() {
        ReservoirSampler sampler1 = new ReservoirSampler();
        ReservoirSampler sampler2 = new ReservoirSampler();
        int totalNumberOfVectors = 100;
        int sampleSize = 10;

        int[] sampledIndices1 = sampler1.sample(totalNumberOfVectors, sampleSize);
        int[] sampledIndices2 = sampler2.sample(totalNumberOfVectors, sampleSize);

        assertNotEquals(Arrays.toString(sampledIndices1), Arrays.toString(sampledIndices2));
    }

    public void testEdgeCaseZeroVectors() {
        ReservoirSampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 0;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(0, sampledIndices.length);
    }

    public void testEdgeCaseZeroSampleSize() {
        ReservoirSampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 10;
        int sampleSize = 0;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(0, sampledIndices.length);
    }
}
