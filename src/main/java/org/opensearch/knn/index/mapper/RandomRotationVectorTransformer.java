/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.quantization.quantizer.RandomGaussianRotation;

/**
 * Normalizes vectors using L2 (Euclidean) normalization, ensuring the vector's
 * magnitude becomes 1 while preserving its directional properties.
 */
@Log4j2
public class RandomRotationVectorTransformer implements VectorTransformer {
    float[][] rotationMatrix;

    public RandomRotationVectorTransformer(int dimension) {
        log.info("making new matrix");
//        this.rotationMatrix = rotationMatrix;
        this.rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimension);
    }

    @Override
    public void transform(float[] vector) {
        validateVector(vector);
        float[] rotatedVector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        System.arraycopy(rotatedVector, 0, vector, 0, vector.length);
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

    private void validateVector(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }
    }
}
