/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

/**
 * Defines operations for transforming vectors in the k-NN search context.
 * Implementations can modify vectors while preserving their dimensional properties
 * for specific use cases such as normalization, scaling, or other transformations.
 */
public interface VectorTransformer {

    /**
     * Transforms a float vector in place.
     *
     * @param vector The input vector to transform (must not be null)
     * @throws IllegalArgumentException if the input vector is null
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

    /**
     * Wraps a {@link KNNFloatVectorValues} stream so that each vector returned by {@code getVector()}
     * is transformed on demand. Default implementation is a pass-through.
     *
     * <p>Used by codec-layer components (e.g. {@code KNN80DocValuesConsumer}) to apply vector
     * transformations to the stream of vectors fed into native index builders, without mutating
     * the original vectors stored in {@code BinaryDocValues}.
     *
     * @param delegate the underlying stream of float vectors
     * @return a stream that applies the transformation on the fly; returns {@code delegate} unchanged
     *         for no-op implementations
     */
    default KNNFloatVectorValues wrap(final KNNFloatVectorValues delegate) {
        return delegate;
    }

    /**
     * Wraps a {@link KNNByteVectorValues} stream. Default implementation is a pass-through.
     * Kept symmetric with {@link #wrap(KNNFloatVectorValues)} so callers can apply transformations
     * uniformly regardless of the vector element type.
     */
    default KNNByteVectorValues wrap(final KNNByteVectorValues delegate) {
        return delegate;
    }
}
