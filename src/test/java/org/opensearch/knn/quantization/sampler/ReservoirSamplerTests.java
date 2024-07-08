/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.sampler;

import org.opensearch.knn.KNNTestCase;


public class ReservoirSamplerTests extends KNNTestCase {

    public void testSample() {
        Sampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 100;
        int sampleSize = 10;

        int[] samples = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(sampleSize, samples.length);
        for (int index : samples) {
            assertTrue(index >= 0 && index < totalNumberOfVectors);
        }
    }

    public void testSample_withFullSampling() {
        Sampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 10;
        int sampleSize = 10;

        int[] samples = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(sampleSize, samples.length);
        for (int index : samples) {
            assertTrue(index >= 0 && index < totalNumberOfVectors);
        }
    }

    public void testSample_withLessVectors() {
        Sampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 5;
        int sampleSize = 10;

        int[] samples = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(totalNumberOfVectors, samples.length);
        for (int index : samples) {
            assertTrue(index >= 0 && index < totalNumberOfVectors);
        }
    }

    public void testSample_withZeroVectors() {
        Sampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 0;
        int sampleSize = 10;

        int[] samples = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(0, samples.length);
    }

    public void testSample_withOneVector() {
        Sampler sampler = new ReservoirSampler();
        int totalNumberOfVectors = 1;
        int sampleSize = 10;

        int[] samples = sampler.sample(totalNumberOfVectors, sampleSize);
        assertEquals(1, samples.length);
        assertTrue(samples[0] == 0);
    }
}

