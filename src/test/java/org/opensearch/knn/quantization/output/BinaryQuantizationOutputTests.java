/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.output;

import org.junit.Before;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;

public class BinaryQuantizationOutputTests extends KNNTestCase {

    private static final int BITS_PER_COORDINATE = 1;
    private BinaryQuantizationOutput quantizationOutput;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        quantizationOutput = new BinaryQuantizationOutput(BITS_PER_COORDINATE);
    }

    public void testPrepareQuantizedVector_ShouldInitializeCorrectly_WhenVectorLengthIsValid() {
        // Arrange
        int vectorLength = 10;

        // Act
        quantizationOutput.prepareQuantizedVector(vectorLength);

        // Assert
        assertNotNull(quantizationOutput.getQuantizedVector());
    }

    public void testPrepareQuantizedVector_ShouldThrowException_WhenVectorLengthIsZeroOrNegative() {
        // Act and Assert
        expectThrows(IllegalArgumentException.class, () -> quantizationOutput.prepareQuantizedVector(0));
        expectThrows(IllegalArgumentException.class, () -> quantizationOutput.prepareQuantizedVector(-1));
    }

    public void testIsPrepared_ShouldReturnTrue_WhenCalledWithSameVectorLength() {
        // Arrange
        int vectorLength = 8;
        quantizationOutput.prepareQuantizedVector(vectorLength);
        // Act and Assert
        assertTrue(quantizationOutput.isPrepared(vectorLength));
    }

    public void testIsPrepared_ShouldReturnFalse_WhenCalledWithDifferentVectorLength() {
        // Arrange
        int vectorLength = 8;
        quantizationOutput.prepareQuantizedVector(vectorLength);
        // Act and Assert
        assertFalse(quantizationOutput.isPrepared(vectorLength + 1));
    }

    public void testGetQuantizedVector_ShouldReturnSameReference() {
        // Arrange
        int vectorLength = 5;
        quantizationOutput.prepareQuantizedVector(vectorLength);
        // Act
        byte[] vector = quantizationOutput.getQuantizedVector();
        // Assert
        assertEquals(vector, quantizationOutput.getQuantizedVector());
    }

    public void testGetQuantizedVectorCopy_ShouldReturnCopyOfVector() {
        // Arrange
        int vectorLength = 5;
        quantizationOutput.prepareQuantizedVector(vectorLength);

        // Act
        byte[] vectorCopy = quantizationOutput.getQuantizedVectorCopy();

        // Assert
        assertNotSame(vectorCopy, quantizationOutput.getQuantizedVector());
        assertArrayEquals(vectorCopy, quantizationOutput.getQuantizedVector());
    }

    public void testGetQuantizedVectorCopy_ShouldReturnNewCopyOnEachCall() {
        // Arrange
        int vectorLength = 5;
        quantizationOutput.prepareQuantizedVector(vectorLength);

        // Act
        byte[] vectorCopy1 = quantizationOutput.getQuantizedVectorCopy();
        byte[] vectorCopy2 = quantizationOutput.getQuantizedVectorCopy();

        // Assert
        assertNotSame(vectorCopy1, vectorCopy2);
    }

    public void testPrepareQuantizedVector_ShouldResetQuantizedVector_WhenCalledWithDifferentLength() {
        // Arrange
        int initialLength = 5;
        int newLength = 10;
        quantizationOutput.prepareQuantizedVector(initialLength);
        byte[] initialVector = quantizationOutput.getQuantizedVector();

        // Act
        quantizationOutput.prepareQuantizedVector(newLength);
        byte[] newVector = quantizationOutput.getQuantizedVector();

        // Assert
        assertNotSame(initialVector, newVector); // The array reference should change
        assertEquals(newVector.length, (BITS_PER_COORDINATE * newLength + 7) / 8); // Correct size for new vector
    }

    public void testPrepareQuantizedVector_ShouldRetainSameArray_WhenCalledWithSameLength() {
        // Arrange
        int vectorLength = 5;
        quantizationOutput.prepareQuantizedVector(vectorLength);
        byte[] initialVector = quantizationOutput.getQuantizedVector();

        // Act
        quantizationOutput.prepareQuantizedVector(vectorLength);
        byte[] newVector = quantizationOutput.getQuantizedVector();

        // Assert
        assertSame(newVector, initialVector); // The array reference should remain the same
    }
}
