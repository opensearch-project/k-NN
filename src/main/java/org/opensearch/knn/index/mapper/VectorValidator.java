/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

/**
 * Class validates vector after it has been parsed
 */
public interface VectorValidator {
    /**
     * Validate if the given byte vector is supported
     *
     * @param vector     the given vector
     */
    default void validateVector(byte[] vector) {}

    /**
     * Validate if the given float vector is supported
     *
     * @param vector     the given vector
     */
    default void validateVector(float[] vector) {}

    VectorValidator NOOP_VECTOR_VALIDATOR = new VectorValidator() {
    };
}
