/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.sampler;

/**
 * SamplingFactory is a factory class for creating instances of Sampler.
 * It uses the factory design pattern to encapsulate the creation logic for different types of samplers.
 */
public final class SamplingFactory {

    /**
     * Private constructor to prevent instantiation of this  class.
     * The class is not meant to be instantiated, as it provides static methods only.
     */
    private SamplingFactory() {

    }

    /**
     * SamplerType is an enumeration of the different types of samplers that can be created by the factory.
     */
    public enum SamplerType {
        RESERVOIR, // Represents a reservoir sampling strategy
        // Add more enum values here for additional sampler types
    }

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
                return new ReservoirSampler();
            // Add more cases for different samplers here
            default:
                throw new IllegalArgumentException("Unsupported sampler type: " + samplerType);
        }
    }
}
