/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;

/**
 * An abstract wrapper around a {@link FloatVectorValues} instance, providing a consistent
 * interface for layered or decorated vector value implementations.
 * <p>
 * This class is typically used when multiple wrappers are stacked around a base
 * {@link FloatVectorValues}, and utility methods are needed to access the innermost
 * (bottom-level) implementation.
 */
@RequiredArgsConstructor
public abstract class WrappedFloatVectorValues extends FloatVectorValues {

    // The wrapped (nested) {@link FloatVectorValues} instance.
    protected final FloatVectorValues floatVectorValues;

    /**
     * Extracts the bottom-level {@link FloatVectorValues} from a possibly wrapped
     * {@link KnnVectorValues} instance.
     * <p>
     * If the provided {@code knnVectorValues} is not an instance of
     * {@link FloatVectorValues}, this method returns {@code null}. Otherwise, it unwraps
     * any nested {@link WrappedFloatVectorValues} layers until it reaches the base
     * {@link FloatVectorValues}.
     *
     * @param knnVectorValues the {@link KnnVectorValues} to unwrap
     * @return the innermost {@link FloatVectorValues}, or {@code null} if not applicable
     */
    public static FloatVectorValues getBottomFloatVectorValues(KnnVectorValues knnVectorValues) {
        if (knnVectorValues instanceof FloatVectorValues floatVectorValues) {
            if (floatVectorValues instanceof WrappedFloatVectorValues wrappedFloatVectorValues) {
                return wrappedFloatVectorValues.floatVectorValues;
            }
            return floatVectorValues;
        }

        return null;
    }
}
