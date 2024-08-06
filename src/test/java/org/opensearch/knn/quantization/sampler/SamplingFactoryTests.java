/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import org.opensearch.knn.KNNTestCase;

public class SamplingFactoryTests extends KNNTestCase {
    public void testGetSampler_withReservoir() {
        Sampler sampler = SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR);
        assertTrue(sampler instanceof ReservoirSampler);
    }

    public void testGetSampler_withUnsupportedType() {
        expectThrows(NullPointerException.class, () -> SamplingFactory.getSampler(null)); // This should throw an exception
    }
}
