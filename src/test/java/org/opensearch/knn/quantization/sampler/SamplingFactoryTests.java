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

public class SamplingFactoryTests extends KNNTestCase {
    public void testGetSampler_withReservoir() {
        Sampler sampler = SamplingFactory.getSampler(SamplingFactory.SamplerType.RESERVOIR);
        assertTrue(sampler instanceof ReservoirSampler);
    }

    public void testGetSampler_withUnsupportedType() {
        expectThrows( NullPointerException.class, ()-> SamplingFactory.getSampler(null)); // This should throw an exception
    }
}
