/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.VectorUtil;

/**
 * Normalizes vectors using L2 (Euclidean) normalization, ensuring the vector's
 * magnitude becomes 1 while preserving its directional properties.
 */
@Log4j2
public class NormalizeVectorTransformer implements VectorTransformer {
    public NormalizeVectorTransformer() {
        log.info("Making normalize vector transformer");
    }

    @Override
    public void transform(float[] vector) {
//        log.info("transform called...");
        validateVector(vector);
        VectorUtil.l2normalize(vector);
    }

    /**
     * Transforms a byte array vector by normalizing it.
     * This operation is currently not supported for byte arrays.
     *
     * @param vector the byte array to be normalized
     * @throws UnsupportedOperationException when this method is called, as byte array normalization is not supported
     */
    @Override
    public void transform(byte[] vector) {
        throw new UnsupportedOperationException("Byte array normalization is not supported");
    }

    @Override
    public void undoTransform(float[] vector) {
        // Normalization is not reversible - this is a no-op
        validateVector(vector);
    }

    private void validateVector(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }
    }
}
