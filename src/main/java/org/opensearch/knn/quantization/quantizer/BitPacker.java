/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.experimental.UtilityClass;

/**
 * The BitPackingUtil class provides utility methods for packing bits into a byte array.
 * This class is designed to be used by quantizers that need to convert floating-point vectors
 * into compact binary representations by comparing them against quantization thresholds.
 *
 * <p>
 * The methods in this class handle both single-bit and multi-bit quantization scenarios,
 * allowing for efficient storage and transmission of quantized vectors.
 * </p>
 *
 * <p>
 * This class is marked as a utility class using Lombok's {@link lombok.experimental.UtilityClass} annotation,
 * making it a singleton and preventing instantiation.
 * </p>
 */
@UtilityClass
class BitPacker {

    /**
     * Packs bits into a byte array based on the provided thresholds and vector.
     *
     * @param vector             the vector to quantize.
     * @param thresholds         the thresholds for quantization.
     * @param bitsPerCoordinate  the number of bits used per coordinate.
     * @return the packed bits as a byte array.
     */
    byte[] packBits(final float[] vector, final float[][] thresholds, final int bitsPerCoordinate) {
        int totalBits = bitsPerCoordinate * vector.length;
        int byteLength = (totalBits + 7) >> 3; // Calculate byte length needed
        byte[] packedBits = new byte[byteLength];

        for (int i = 0; i < bitsPerCoordinate; i++) {
            for (int j = 0; j < vector.length; j++) {
                if (vector[j] > thresholds[i][j]) {
                    int bitPosition = i * vector.length + j;
                    int byteIndex = bitPosition >> 3; // Equivalent to bitPosition / 8
                    int bitIndex = 7 - (bitPosition & 7); // Equivalent to 7 - (bitPosition % 8)
                    packedBits[byteIndex] |= (1 << bitIndex); // Set the bit
                }
            }
        }

        return packedBits;
    }

    /**
     * Overloaded method to pack bits for one-bit quantization.
     *
     * @param vector             the vector to quantize.
     * @param thresholds         the thresholds for quantization.
     * @return the packed bits as a byte array.
     */
    byte[] packBits(final float[] vector, final float[] thresholds) {
        return packBits(vector, new float[][] { thresholds }, 1);
    }
}
