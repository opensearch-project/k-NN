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

public class SamplingFactory {
    public enum SamplerType {
        RESERVOIR,
    }

    public static Sampler getSampler(SamplerType samplerType) {
        switch (samplerType) {
            case RESERVOIR:
                return new ReservoirSampler();
            // Add more cases for different samplers
            default:
                throw new IllegalArgumentException("Unsupported sampler type: " + samplerType);
        }
    }
}
