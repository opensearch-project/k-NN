/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;

public class BitPackerTests extends KNNTestCase {

    public void testSingleBitQuantization_whenValuesAboveThreshold_thenSetsCorrespondingBits() {
        float[] vector = { 1.2f, 3.4f, 5.6f };
        float[] thresholds = { 2.0f, 3.0f, 4.0f };
        byte[] packedBits = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, packedBits);

        // coordinates at index 1 and 2 exceed thresholds -> bits at positions 1 and 2
        assertEquals((byte) 0x60, packedBits[0]);
    }

    public void testSingleBitQuantization_whenNoValuesAboveThreshold_thenAllBitsZero() {
        float[] vector = { 0.1f, 0.2f };
        float[] thresholds = { 1.0f, 2.0f };
        byte[] packedBits = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, packedBits);

        assertEquals((byte) 0x00, packedBits[0]);
    }

    public void testSingleBitQuantization_whenAllValuesAboveThreshold_thenAllBitsSet() {
        float[] vector = { 3.0f, 4.0f, 5.0f };
        float[] thresholds = { 1.0f, 2.0f, 3.0f };
        byte[] packedBits = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, packedBits);

        // all three coordinates set bits at positions 0, 1, 2 -> top 3 bits of byte
        assertEquals((byte) 0xE0, packedBits[0]);
    }

    public void testSingleBitQuantization_whenEqualToThreshold_thenBitNotSet() {
        float[] vector = { 2.0f, 3.0f };
        float[] thresholds = { 2.0f, 3.0f };
        byte[] packedBits = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, packedBits);

        // comparison is strictly greater than threshold
        assertEquals((byte) 0x00, packedBits[0]);
    }

    public void testMultiBitQuantization_whenMixedThresholdResults_thenPacksCorrectly() {
        float[] vector = { 1.2f, 3.4f, 5.6f };
        float[][] thresholds = { { 1.0f, 3.0f, 5.0f }, { 1.5f, 3.5f, 5.5f } };
        byte[] packedBits = new byte[2];

        BitPacker.quantizeAndPackBits(vector, thresholds, 2, packedBits);

        // first bit plane: all coordinates exceed; second bit plane: only index 2 exceeds at bit position 5
        assertEquals((byte) 0xE4, packedBits[0]);
        assertEquals((byte) 0x00, packedBits[1]);
    }

    public void testMultiBitQuantization_whenVectorSpansMultipleBytes_thenPacksAcrossBytes() {
        int vectorLength = 10;
        float[] vector = new float[vectorLength];
        float[][] thresholds = new float[1][vectorLength];
        for (int i = 0; i < vectorLength; i++) {
            vector[i] = 2.0f;
            thresholds[0][i] = 1.0f;
        }
        byte[] packedBits = new byte[2];

        BitPacker.quantizeAndPackBits(vector, thresholds, 1, packedBits);

        // all 10 bits set in first byte (8 bits) and second byte (2 bits)
        assertEquals((byte) 0xFF, packedBits[0]);
        assertEquals((byte) 0xC0, packedBits[1]);
    }

    public void testMultiBitQuantization_whenFourBitsPerCoordinate_thenUsesAllBitPlanes() {
        float[] vector = { 5.0f, 5.0f };
        float[][] thresholds = { { 1.0f, 1.0f }, { 2.0f, 2.0f }, { 3.0f, 3.0f }, { 4.0f, 4.0f } };
        byte[] packedBits = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, 4, packedBits);

        // four bit planes, two coordinates each -> 8 bits all set
        assertEquals((byte) 0xFF, packedBits[0]);
    }

    public void testMultiBitQuantization_singleBitOverloadDelegatesToMultiBit() {
        float[] vector = { 3.0f };
        float[] thresholds = { 2.0f };
        byte[] singleBitPacked = new byte[1];
        byte[] multiBitPacked = new byte[1];

        BitPacker.quantizeAndPackBits(vector, thresholds, singleBitPacked);
        BitPacker.quantizeAndPackBits(vector, new float[][] { thresholds }, 1, multiBitPacked);

        assertEquals(singleBitPacked[0], multiBitPacked[0]);
    }
}
