/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteOrder;

public class FaissBF16ReconstructorTests extends KNNTestCase {

    public void testReconstruct_basicValues() {
        int dimension = 4;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);

        // BF16 stores the upper 16 bits of a float32.
        float[] testValues = { 1.0f, -1.0f, 0.0f, 2.0f };
        byte[] quantizedBytes = encode(testValues, ByteOrder.nativeOrder());

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        // BF16 reconstruction should produce exact values for these simple floats
        // (since they don't lose precision in the truncation)
        for (int i = 0; i < dimension; i++) {
            assertEquals("Mismatch at index " + i, testValues[i], result[i], 0.0f);
        }
    }

    public void testReconstruct_precisionLoss() {
        int dimension = 1;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);

        float original = 1.1f;
        int floatBits = Float.floatToIntBits(original);
        int bf16Bits = (floatBits >> 16) & 0xFFFF;

        byte[] quantizedBytes = encode(new float[] { original }, ByteOrder.nativeOrder());

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        // The result should be close but not exact due to BF16 precision loss
        float expected = Float.intBitsToFloat(bf16Bits << 16);
        assertEquals(expected, result[0], 0.0f);

        // Should be within ~1% of original
        assertEquals(original, result[0], 0.02f);
    }

    public void testReconstruct_littleEndian_viaInjectedByteOrder() {
        // Directly construct with LITTLE_ENDIAN — covers the `byteOrder == LITTLE_ENDIAN` branch (true).
        int dimension = 4;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16, ByteOrder.LITTLE_ENDIAN);

        float[] testValues = { 1.0f, -1.0f, 0.0f, 2.0f };
        byte[] quantizedBytes = encode(testValues, ByteOrder.LITTLE_ENDIAN);

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        for (int i = 0; i < dimension; i++) {
            assertEquals("Mismatch at index " + i, testValues[i], result[i], 0.0f);
        }
    }

    public void testReconstruct_bigEndian_viaInjectedByteOrder() {
        // Directly construct with BIG_ENDIAN — covers the `byteOrder == LITTLE_ENDIAN` branch (false),
        // which is the platform-dependent partial branch that reflection-only tests couldn't reach.
        int dimension = 4;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16, ByteOrder.BIG_ENDIAN);

        float[] testValues = { 1.0f, -1.0f, 0.0f, 3.5f };
        byte[] quantizedBytes = encode(testValues, ByteOrder.BIG_ENDIAN);

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        for (int i = 0; i < dimension; i++) {
            assertEquals("Mismatch at index " + i, testValues[i], result[i], 0.0f);
        }
    }

    /** Encode float values as BF16 in the specified byte order. */
    private static byte[] encode(float[] values, ByteOrder order) {
        byte[] out = new byte[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            int floatBits = Float.floatToIntBits(values[i]);
            int bf16Bits = (floatBits >> 16) & 0xFFFF;
            if (order == ByteOrder.LITTLE_ENDIAN) {
                out[i * 2] = (byte) (bf16Bits & 0xFF);
                out[i * 2 + 1] = (byte) ((bf16Bits >> 8) & 0xFF);
            } else {
                out[i * 2] = (byte) ((bf16Bits >> 8) & 0xFF);
                out[i * 2 + 1] = (byte) (bf16Bits & 0xFF);
            }
        }
        return out;
    }
}
