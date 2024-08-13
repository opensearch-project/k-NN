/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * SamplingFactory is a factory class for creating instances of Sampler.
 * It uses the factory design pattern to encapsulate the creation logic for different types of samplers.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class SamplingFactory {

    /**
     * Creates and returns a Sampler instance based on the specified SamplerType.
     *
     * @param samplerType the type of sampler to create.
     * @return a Sampler instance.
     * @throws IllegalArgumentException if the sampler type is not supported.
     */
    public static Sampler getSampler(final SamplerType samplerType) {
        switch (samplerType) {
            case RESERVOIR:
                return ReservoirSampler.getInstance();
            // Add more cases for different samplers here
            default:
                throw new IllegalArgumentException("Unsupported sampler type: " + samplerType);
        }
    }
}
