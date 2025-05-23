/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.experimental.UtilityClass;

import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED;

@UtilityClass
public class RandomGaussianRotation {

    /**
     * Generates a random rotation matrix. Each entry is sampled from a standard Gaussian distribution.
     * Then Gram-Schmidt is used to make the random matrix orthonormal (so it represents
     * a rotation or rotation + reflection). The matrix preserves distances with probability 1; norm(Mx) = norm(x)
     * for l1 and l2 norms.
     *
     * Random rotation improves k-NN search by making each vector coordinate roughly equal. That is, it smooths the
     * data so each dimension has roughly equal variance in our vector population. To see this, note that
     * Var[(Mx)_i] = (1/d) sum_j Var[x_j] for each i due to each entry in the random matrix being independent.
     *
     * The RNG is seeded with QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED to achieve reproducible rotations across
     * different indexing runs.
     *
     * @param dimensions The number of dimensions for the rotation matrix.
     * @return A 2D float array representing the rotation matrix.
     */
    public float[][] generateRotationMatrix(int dimensions) {
        Random random = new Random(QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED);
        float[][] rotationMatrix = new float[dimensions][dimensions];

        // Step 1: Generate random Gaussian values
        for (int i = 0; i < dimensions; i++) {
            for (int j = 0; j < dimensions; j++) {
                rotationMatrix[i][j] = (float) random.nextGaussian();
            }
        }

        // Step 2: Orthogonalize the matrix using the Gram-Schmidt process
        for (int i = 0; i < dimensions; i++) {
            // Normalize the current vector
            float norm = 0f;
            for (int j = 0; j < dimensions; j++) {
                norm += rotationMatrix[i][j] * rotationMatrix[i][j];
            }
            norm = (float) Math.sqrt(norm);
            for (int j = 0; j < dimensions; j++) {
                rotationMatrix[i][j] /= norm;
            }

            // Subtract projections of the current vector onto all previous vectors
            for (int k = 0; k < i; k++) {
                float dotProduct = 0f;
                for (int j = 0; j < dimensions; j++) {
                    dotProduct += rotationMatrix[i][j] * rotationMatrix[k][j];
                }
                for (int j = 0; j < dimensions; j++) {
                    rotationMatrix[i][j] -= dotProduct * rotationMatrix[k][j];
                }
            }

            // Re-normalize after orthogonalization
            norm = 0f;
            for (int j = 0; j < dimensions; j++) {
                norm += rotationMatrix[i][j] * rotationMatrix[i][j];
            }
            norm = (float) Math.sqrt(norm);
            for (int j = 0; j < dimensions; j++) {
                rotationMatrix[i][j] /= norm;
            }
        }

        return rotationMatrix;
    }

    /**
     * Applies a rotation to a vector using the provided rotation matrix.
     *
     * @param vector The input vector to be rotated. The input vector is not modified.
     * @param rotationMatrix The rotation matrix.
     * @return The copy of the original vector but rotated.
     */
    public float[] applyRotation(float[] vector, float[][] rotationMatrix) {
        int dimensions = vector.length;
        float[] rotatedVector = new float[dimensions];

        for (int i = 0; i < dimensions; i++) {
            rotatedVector[i] = 0f;
            for (int j = 0; j < dimensions; j++) {
                rotatedVector[i] += rotationMatrix[i][j] * vector[j];
            }
        }

        return rotatedVector;
    }
}
