/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;

import java.util.HashSet;
import java.util.Set;

public class RandomGaussianRotationTests extends KNNTestCase {

    public void testGenerateRotationMatrix_Orthogonality() {
        int dimensions = 5;

        // Generate the rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimensions);

        // Validate dimensions
        assertEquals(dimensions, rotationMatrix.length);
        for (float[] row : rotationMatrix) {
            assertEquals(dimensions, row.length);
        }

        // Validate orthogonality: dot product of distinct rows should be close to zero
        float delta = 0.0001f;
        for (int i = 0; i < dimensions; i++) {
            for (int j = i + 1; j < dimensions; j++) {
                float dotProduct = 0f;
                for (int k = 0; k < dimensions; k++) {
                    dotProduct += rotationMatrix[i][k] * rotationMatrix[j][k];
                }
                assertEquals("Dot product of row " + i + " and row " + j + " is not zero", 0.0f, dotProduct, delta);
            }
        }

        // Validate normalization: length of each row vector should be close to 1
        for (int i = 0; i < dimensions; i++) {
            float norm = 0f;
            for (int j = 0; j < dimensions; j++) {
                norm += rotationMatrix[i][j] * rotationMatrix[i][j];
            }
            assertEquals("Row " + i + " is not normalized", 1.0f, (float) Math.sqrt(norm), delta);
        }
    }

    public void testApplyRotation() {
        float[] vector = { 1.0f, 0.0f, 0.0f };
        int dimensions = vector.length;

        // Generate a rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimensions);

        // Apply the rotation
        float[] rotatedVector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);

        // Validate dimensions
        assertEquals(dimensions, rotatedVector.length);

        // Validate that the rotated vector is non-zero
        float norm = 0f;
        for (float value : rotatedVector) {
            norm += value * value;
        }
        assertTrue("Rotated vector should not be zero", norm > 0);

        // Validate that the rotated vector lies in the same dimensional space
        Set<Integer> nonZeroIndices = new HashSet<>();
        for (int i = 0; i < dimensions; i++) {
            if (Math.abs(rotatedVector[i]) > 0.0001f) {
                nonZeroIndices.add(i);
            }
        }
        assertFalse("Rotated vector contains invalid values", nonZeroIndices.isEmpty());
    }

    public void testOrthogonalityOfGeneratedMatrixWithLargerDimensions() {
        int dimensions = 10;

        // Generate the rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimensions);

        // Validate orthogonality: dot product of distinct rows should be close to zero
        float delta = 0.0001f;
        for (int i = 0; i < dimensions; i++) {
            for (int j = i + 1; j < dimensions; j++) {
                float dotProduct = 0f;
                for (int k = 0; k < dimensions; k++) {
                    dotProduct += rotationMatrix[i][k] * rotationMatrix[j][k];
                }
                assertEquals("Dot product of row " + i + " and row " + j + " is not zero", 0.0f, dotProduct, delta);
            }
        }
    }

    public void testRotationMatrixCorrectness() {
        float[] vector = { 3.0f, 4.0f };
        int dimensions = vector.length;

        // Generate the rotation matrix
        float[][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(dimensions);

        // Apply the rotation
        float[] rotatedVector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);

        // Ensure rotated vector length matches original vector length
        float originalNorm = 0f;
        for (float value : vector) {
            originalNorm += value * value;
        }

        float rotatedNorm = 0f;
        for (float value : rotatedVector) {
            rotatedNorm += value * value;
        }

        assertEquals(
            "Rotated vector norm does not match original vector norm",
            (float) Math.sqrt(originalNorm),
            (float) Math.sqrt(rotatedNorm),
            0.0001f
        );
    }
}
