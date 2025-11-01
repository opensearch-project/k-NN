/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

/**
 * Defines operations for transforming vectors in the k-NN search context.
 * Implementations can modify vectors while preserving their dimensional properties
 * for specific use cases such as normalization, scaling, or other transformations.
 */
public interface VectorTransformer {

    /**
     * Transforms the given vector and returns the resulting vector.
     * <p>
     * Implementations may apply normalization, quantization, or other
     * transformations. When {@code inplaceUpdate} is true, the implementation
     * must perform the transformation directly on the provided input array and
     * return the same reference. When {@code inplaceUpdate} is false, the
     * implementation must create and return a new array to preserve the original
     * input.
     * <p>
     * The default implementation performs no transformation and simply returns
     * the original vector.
     *
     * @param vector        the float array representing a vector; must not be null
     * @param inplaceUpdate if true, the transformation must modify and return the
     *                      original input array; if false, the implementation must
     *                      return a new array instance
     * @return the transformed vector (never null)
     * @throws IllegalArgumentException if vector is null
     */
    default float[] transform(final float[] vector, final boolean inplaceUpdate) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        return vector;
    }

    /**
     * Transforms a byte vector in place.
     *
     * @param vector The input vector to transform (must not be null)
     * @throws IllegalArgumentException if the input vector is null
     */
    default void transform(final byte[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
    }
}
