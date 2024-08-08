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
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 5;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        int[] expectedIndices = IntStream.range(0, totalNumberOfVectors).toArray();
        assertArrayEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleEqualToSampleSize() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 10;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        int[] expectedIndices = IntStream.range(0, totalNumberOfVectors).toArray();
        assertArrayEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleRandomness() {
        ReservoirSampler sampler1 = ReservoirSampler.getInstance();
        ReservoirSampler sampler2 = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 100;
        int sampleSize = 10;

        int[] sampledIndices1 = sampler1.sample(totalNumberOfVectors, sampleSize);
        int[] sampledIndices2 = sampler2.sample(totalNumberOfVectors, sampleSize);

        // It's unlikely but possible for the two samples to be equal, so we just check they are sorted correctly
        Arrays.sort(sampledIndices1);
        Arrays.sort(sampledIndices2);
        assertFalse("Sampled indices should be different", Arrays.equals(sampledIndices1, sampledIndices2));
    }

    public void testEdgeCaseZeroVectors() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 0;
        int sampleSize = 10;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals("Sampled indices should be empty when there are zero vectors.", 0, sampledIndices.length);
    }

    public void testEdgeCaseZeroSampleSize() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 10;
        int sampleSize = 0;
        int[] sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals("Sampled indices should be empty when sample size is zero.", 0, sampledIndices.length);
    }
}