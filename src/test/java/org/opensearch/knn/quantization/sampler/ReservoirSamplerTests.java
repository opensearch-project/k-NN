/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import org.opensearch.knn.KNNTestCase;

import java.util.BitSet;

public class ReservoirSamplerTests extends KNNTestCase {

    public void testSampleLessThanSampleSize() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 5;
        int sampleSize = 10;
        BitSet sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        BitSet expectedIndices = new BitSet(totalNumberOfVectors);
        expectedIndices.set(0, totalNumberOfVectors);
        assertEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleEqualToSampleSize() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 10;
        int sampleSize = 10;
        BitSet sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        BitSet expectedIndices = new BitSet(totalNumberOfVectors);
        expectedIndices.set(0, totalNumberOfVectors);
        assertEquals("Sampled indices should include all available indices.", expectedIndices, sampledIndices);
    }

    public void testSampleRandomness() {
        ReservoirSampler sampler1 = ReservoirSampler.getInstance();
        ReservoirSampler sampler2 = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 100;
        int sampleSize = 10;

        BitSet sampledIndices1 = sampler1.sample(totalNumberOfVectors, sampleSize);
        BitSet sampledIndices2 = sampler2.sample(totalNumberOfVectors, sampleSize);

        assertNotEquals(sampledIndices1, sampledIndices2);
    }

    public void testEdgeCaseZeroVectors() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 0;
        int sampleSize = 10;
        BitSet sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(0, sampledIndices.cardinality());
    }

    public void testEdgeCaseZeroSampleSize() {
        ReservoirSampler sampler = ReservoirSampler.getInstance();
        int totalNumberOfVectors = 10;
        int sampleSize = 0;
        BitSet sampledIndices = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(0, sampledIndices.cardinality());
    }
}
