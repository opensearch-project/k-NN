/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.quantization.quantizer.RandomGaussianRotation;

/**
 * Normalizes vectors using L2 (Euclidean) normalization, ensuring the vector's
 * magnitude becomes 1 while preserving its directional properties.
 */
public class RandomRotationNormalizeVectorTransformer implements VectorTransformer {
    float[][] rotationMatrix;

    // which comes first, normalization or rotation?
    // TODO: review w a few other people around transformer state/matrix state. (Jack, Vijay, Tejas).
    public RandomRotationNormalizeVectorTransformer(int dimension) {
//        this.rotationMatrix = rotationMatrix;
        rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimension);
    }

    @Override
    public void transform(float[] vector) {
        validateVector(vector);
        VectorUtil.l2normalize(vector);
        float[] rotatedVector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        System.arraycopy(vector, 0, rotatedVector, 0, vector.length);
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
        throw new UnsupportedOperationException("Byte array rotation is not supported");
    }

    private void validateVector(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vector cannot be null or empty");
        }
    }
}
