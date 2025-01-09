/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

/**
 * Defines operations for transforming vectors in the k-NN search context.
 * Implementations can modify vectors while preserving their dimensional properties
 * for specific use cases such as normalization, scaling, or other transformations.
 *
 * <p>This interface provides default implementations that pass through the original
 * vector without modification. Implementing classes should override these methods
 * to provide specific transformation logic.
 */
public interface VectorTransformer {

    /**
     * Transforms a float vector into a new vector of the same type.
     * The default implementation returns the input vector unchanged.
     *
     * @param vector The input vector to transform
     * @return The transformed vector
     */
    default float[] transform(float[] vector) {
        return vector;
    }

    /**
     * Transforms a byte vector into a new vector of the same type.
     * The default implementation returns the input vector unchanged.
     *
     * @param vector The input vector to transform
     * @return The transformed vector
     */
    default byte[] transform(byte[] vector) {
        return vector;
    }

    /**
     * A no-operation transformer that returns vectors unchanged.
     * This constant can be used when no transformation is needed.
     */
    VectorTransformer NOOP_VECTOR_TRANSFORMER = new VectorTransformer() {
    };
}
