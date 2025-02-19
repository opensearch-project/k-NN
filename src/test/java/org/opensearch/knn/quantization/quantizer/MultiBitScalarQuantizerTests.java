/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;

import java.io.IOException;

public class MultiBitScalarQuantizerTests extends KNNTestCase {

    public void testTrain_twoBit() throws IOException {
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        QuantizationState state = twoBitQuantizer.train(request);

        assertTrue(state instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState mbState = (MultiBitScalarQuantizationState) state;
        assertNotNull(mbState.getThresholds());
        assertEquals(2, mbState.getThresholds().length); // 2-bit quantization should have 2 thresholds
    }

    public void testTrain_fourBit() throws IOException {
        MultiBitScalarQuantizer fourBitQuantizer = new MultiBitScalarQuantizer(4);
        float[][] vectors = {
            { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        QuantizationState state = fourBitQuantizer.train(request);

        assertTrue(state instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState mbState = (MultiBitScalarQuantizationState) state;
        assertNotNull(mbState.getThresholds());
        assertEquals(4, mbState.getThresholds().length); // 4-bit quantization should have 4 thresholds
    }

    public void testQuantize_twoBit() throws IOException {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = { { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);
        twoBitQuantizer.quantize(vector, state, output);

        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_fourBit() throws IOException {
        MultiBitScalarQuantizer fourBitQuantizer = new MultiBitScalarQuantizer(4);
        float[] vector = { 1.3f, 2.2f, 3.3f, 4.1f, 5.6f, 6.7f, 7.4f, 8.1f };
        float[][] thresholds = {
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f },
            { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f },
            { 1.2f, 2.2f, 3.2f, 4.2f, 5.2f, 6.2f, 7.2f, 8.2f },
            { 1.3f, 2.3f, 3.3f, 4.3f, 5.3f, 6.3f, 7.3f, 8.3f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(4);
        fourBitQuantizer.quantize(vector, state, output);

        assertNotNull(output.getQuantizedVector());
    }

    public void testQuantize_withNullVector() throws IOException {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);
        output.prepareQuantizedVector(8); // Example length
        expectThrows(
            IllegalArgumentException.class,
            () -> twoBitQuantizer.quantize(
                null,
                new MultiBitScalarQuantizationState(new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT), new float[2][8]),
                output
            )
        );
    }

    public void testQuantize_twoBit_multiple_times() throws IOException {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { -2.5f, 1.5f, -0.5f, 4.0f, 6.5f, -3.5f, 0.0f, 7.2f };
        float[][] thresholds = {
            { -3.0f, 1.0f, -1.0f, 3.5f, 5.0f, -4.0f, 0.5f, 7.0f },
            { -2.0f, 2.0f, 0.0f, 4.5f, 6.0f, -2.5f, -0.5f, 8.0f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);

        // First quantization
        twoBitQuantizer.quantize(vector, state, output);
        assertNotNull(output.getQuantizedVector());

        // Save the reference to the byte array
        byte[] firstByteArray = output.getQuantizedVector();

        // Expected output after the first quantization
        byte[] expectedPackedBits = new byte[] { (byte) 0b11111101, (byte) 0b00001010 };

        // Check the output value after the first quantization
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());

        // Modify vector for a second quantization call
        vector = new float[] { -2.5f, 1.5f, -0.5f, 4.0f, 6.5f, -3.5f, 0.0f, 7.2f };

        // Second quantization
        twoBitQuantizer.quantize(vector, state, output);

        // Assert that the same byte array reference is used
        assertSame(firstByteArray, output.getQuantizedVector());

        // Expected output after the second quantization (based on updated vector)
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_ReuseByteArray_forMultiBitQuantizer() throws IOException {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[] vector = { -2.5f, 1.5f, -0.5f, 4.0f, 6.5f, -3.5f, 0.0f, 7.2f };
        float[][] thresholds = {
            { -3.0f, 1.0f, -1.0f, 3.5f, 5.0f, -4.0f, 0.5f, 7.0f },
            { -2.0f, 2.0f, 0.0f, 4.5f, 6.0f, -2.5f, -0.5f, 8.0f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);

        // First quantization
        twoBitQuantizer.quantize(vector, state, output);
        byte[] firstByteArray = output.getQuantizedVector();

        // Expected output after the first quantization
        byte[] expectedPackedBits = new byte[] { (byte) 0b11111101, (byte) 0b00001010 };

        // Check the output value after the first quantization
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());

        // Second quantization with the same vector length
        twoBitQuantizer.quantize(vector, state, output);
        byte[] secondByteArray = output.getQuantizedVector();

        // Assert that the same byte array reference is used
        assertSame(firstByteArray, secondByteArray);

        // Check the output value after the second quantization
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());

        // Third quantization with the same vector length
        twoBitQuantizer.quantize(vector, state, output);
        byte[] thirdByteArray = output.getQuantizedVector();

        // Assert that the same byte array reference is still used
        assertSame(firstByteArray, thirdByteArray);

        // Check the output value after the third quantization
        assertArrayEquals(expectedPackedBits, output.getQuantizedVector());
    }

    public void testQuantize_withMultipleVectors_inLoop() throws IOException {
        MultiBitScalarQuantizer twoBitQuantizer = new MultiBitScalarQuantizer(2);
        float[][] vectors = {
            { -2.5f, 1.5f, -0.5f, 4.0f, 6.5f, -3.5f, 0.0f, 7.2f },
            { 2.0f, -1.0f, 3.5f, 0.0f, 5.5f, -2.5f, 1.5f, 6.0f },
            { -4.0f, 2.0f, -1.5f, 3.5f, -0.5f, 1.0f, 2.5f, -3.0f } };
        float[][] thresholds = {
            { -3.0f, 1.0f, -1.0f, 3.5f, 5.0f, -4.0f, 0.5f, 7.0f },
            { -2.0f, 2.0f, 0.0f, 4.5f, 6.0f, -2.5f, -0.5f, 8.0f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);

        BinaryQuantizationOutput output = new BinaryQuantizationOutput(2);

        byte[] previousByteArray = null;
        for (float[] vector : vectors) {
            // Check if output is already prepared before quantization
            boolean wasPrepared = output.isPrepared(vector.length);

            // Prepare the output for the new vector length
            output.prepareQuantizedVector(vector.length);

            // Ensure that if it was prepared, it stays the same reference
            if (wasPrepared) {
                assertSame(previousByteArray, output.getQuantizedVector());
            }

            // Perform the quantization
            twoBitQuantizer.quantize(vector, state, output);

            // Save the reference to the byte array after quantization
            previousByteArray = output.getQuantizedVector();

            // Check that the output vector is correctly prepared
            assertTrue(output.isPrepared(vector.length));
        }
    }

    // Mock classes for testing
    private static class MockTrainingRequest extends TrainingRequest<float[]> {
        private final float[][] vectors;

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors) {
            super(vectors.length);
            this.vectors = vectors;
        }

        @Override
        public float[] getVectorAtThePosition(int position) {
            return vectors[position];
        }

        @Override
        public void resetVectorValues() {
            // No-op
        }
    }
}
