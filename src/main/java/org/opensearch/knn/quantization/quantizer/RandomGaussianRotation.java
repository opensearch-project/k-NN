/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.experimental.UtilityClass;

import java.util.Random;

@UtilityClass
public class RandomGaussianRotation {

    /**
     * Generates a random rotation matrix using Gaussian distribution and orthogonalization.
     *
     * @param dimensions The number of dimensions for the rotation matrix.
     * @return A 2D float array representing the rotation matrix.
     */
    public float[][] generateRotationMatrix(int dimensions) {
        Random random = new Random();
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
     * @param vector The input vector to be rotated.
     * @param rotationMatrix The rotation matrix.
     * @return The rotated vector.
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
