/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.util.VectorUtil;

/**
 * Normalizes vectors using L2 (Euclidean) normalization. This transformation ensures
 * that the vector's magnitude becomes 1 while preserving its directional properties.
 */
public class NormalizeVectorTransformer implements VectorTransformer {

    /**
     * Transforms the input vector into unit vector by applying L2 normalization.
     *
     * @param vector The input vector to be normalized. Must not be null.
     * @return A new float array containing the L2-normalized version of the input vector.
     *         Each component is divided by the Euclidean norm of the vector.
     * @throws IllegalArgumentException if the input vector is null, empty, or a zero vector
     */
    @Override
    public float[] transform(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }
        return VectorUtil.l2normalize(vector);
    }
}
