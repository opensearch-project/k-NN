/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.util.Arrays;

/**
 * Defines operations for transforming vectors in the k-NN search context.
 * Implementations can modify vectors while preserving their dimensional properties
 * for specific use cases such as normalization, scaling, or other transformations.
 */
public interface VectorTransformer {

    /**
     * Transforms a float vector into a new vector of the same type.
     *
     * Example:
     * <pre>{@code
     * float[] input = {1.0f, 2.0f, 3.0f};
     * float[] transformed = transformer.transform(input);
     * }</pre>
     *
     * @param vector The input vector to transform (must not be null)
     * @return The transformed vector
     * @throws IllegalArgumentException if the input vector is null
     */
    default float[] transform(final float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        return Arrays.copyOf(vector, vector.length);
    }

    /**
     * Transforms a byte vector into a new vector of the same type.
     *
     * Example:
     * <pre>{@code
     * byte[] input = {1, 2, 3};
     * byte[] transformed = transformer.transform(input);
     * }</pre>
     *
     * @param vector The input vector to transform (must not be null)
     * @return The transformed vector
     * @throws IllegalArgumentException if the input vector is null
     */
    default byte[] transform(final byte[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        // return copy of vector to avoid side effects
        return Arrays.copyOf(vector, vector.length);

    }

    /**
     * A no-operation transformer that returns vector values unchanged.
     * This constant can be used when no transformation is needed.
     */
    VectorTransformer NOOP_VECTOR_TRANSFORMER = new VectorTransformer() {
    };
}
