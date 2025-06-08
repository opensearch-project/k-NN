/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import lombok.experimental.UtilityClass;

/**
 * The BitPacker class provides utility methods for quantizing floating-point vectors and packing the resulting bits
 * into a pre-allocated byte array. This class supports both single-bit and multi-bit quantization scenarios,
 * enabling efficient storage and transmission of quantized vectors.
 *
 * <p>
 * The methods in this class are designed to be used by quantizers that need to convert floating-point vectors
 * into compact binary representations by comparing them against quantization thresholds.
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
     * Quantizes a given floating-point vector and packs the resulting quantized bits into a provided byte array.
     * This method operates by comparing each element of the input vector against corresponding thresholds
     * and encoding the results into a compact binary format using the specified number of bits per coordinate.
     *
     * <p>
     * The method supports multi-bit quantization where each coordinate of the input vector can be represented
     * by multiple bits. For example, with 2-bit quantization, each coordinate is encoded into 2 bits, allowing
     * for four distinct levels of quantization per coordinate.
     * </p>
     *
     * <p>
     * <b>Example:</b>
     * </p>
     * <p>
     * Consider a vector with 3 coordinates: <code>[1.2, 3.4, 5.6]</code> and thresholds:
     * </p>
     * <pre>
     * thresholds = {
     *     {1.0, 3.0, 5.0},  // First bit thresholds
     *     {1.5, 3.5, 5.5}   // Second bit thresholds
     * };
     * </pre>
     * <p>
     * If the number of bits per coordinate is 2, the quantization process will proceed as follows:
     * </p>
     * <ul>
     *     <li>First bit comparison:
     *         <ul>
     *             <li>1.2 > 1.0 -> 1</li>
     *             <li>3.4 > 3.0 -> 1</li>
     *             <li>5.6 > 5.0 -> 1</li>
     *         </ul>
     *     </li>
     *     <li>Second bit comparison:
     *         <ul>
     *             <li>1.2 <= 1.5 -> 0</li>
     *             <li>3.4 <= 3.5 -> 0</li>
     *             <li>5.6 > 5.5 -> 1</li>
     *         </ul>
     *     </li>
     * </ul>
     * <p>
     * The resulting quantized bits will be <code>11 10 11</code>, which is packed into the provided byte array.
     * If there are fewer than 8 bits, the remaining bits in the byte are set to 0.
     * </p>
     *
     * <p>
     * <b>Packing Process:</b>
     * The quantized bits are packed into the byte array. The first coordinate's bits are stored in the most
     * significant positions of the first byte, followed by the second coordinate, and so on. In the example
     * above, the resulting byte array will have the following binary representation:
     * </p>
     * <pre>
     * packedBits = [11011000] // Only the first 6 bits are used, and the last two are set to 0.
     * </pre>
     *
     * <p><b>Bitwise Operations Explanation:</b></p>
     * <ul>
     *     <li><b>byteIndex:</b> This is calculated using <code>byteIndex = bitPosition >> 3</code>, which is equivalent to <code>bitPosition / 8</code>. It determines which byte in the byte array the current bit should be placed in.</li>
     *     <li><b>bitIndex:</b> This is calculated using <code>bitIndex = 7 - (bitPosition & 7)</code>, which is equivalent to <code>7 - (bitPosition % 8)</code>. It determines the exact bit position within the byte.</li>
     *     <li><b>Setting the bit:</b> The bit is set using <code>packedBits[byteIndex] |= (1 << bitIndex)</code>. This shifts a 1 into the correct bit position and ORs it with the existing byte value to set the bit.</li>
     * </ul>
     *
     * @param vector             the floating-point vector to be quantized.
     * @param thresholds         a 2D array representing the quantization thresholds. The first dimension corresponds to the number of bits per coordinate, and the second dimension corresponds to the vector's length.
     * @param bitsPerCoordinate  the number of bits used per coordinate, determining the granularity of the quantization.
     * @param packedBits         the byte array where the quantized bits will be packed.
     */
    void quantizeAndPackBits(final float[] vector, final float[][] thresholds, final int bitsPerCoordinate, byte[] packedBits) {
        int vectorLength = vector.length;
        for (int i = 0; i < bitsPerCoordinate; i++) {
            for (int j = 0; j < vectorLength; j++) {
                if (vector[j] > thresholds[i][j]) {
                    int bitPosition = i * vectorLength + j;
                    // Calculate the index of the byte in the packedBits array.
                    int byteIndex = bitPosition >> 3; // Equivalent to bitPosition / 8
                    // Calculate the bit index within the byte.
                    int bitIndex = 7 - (bitPosition & 7); // Equivalent to 7 - (bitPosition % 8)
                    // Set the bit at the calculated position.
                    packedBits[byteIndex] |= (1 << bitIndex); // Set the bit at bitIndex
                }
            }
        }
    }

    /**
     * Overloaded method to quantize a vector using single-bit quantization and pack the results into a provided byte array.
     *
     * <p>
     * This method is specifically designed for one-bit quantization scenarios, where each coordinate of the
     * vector is represented by a single bit indicating whether the value is above or below the threshold.
     * </p>
     *
     * <p><b>Example:</b></p>
     * <p>
     * If we have a vector <code>[1.2, 3.4, 5.6]</code> and thresholds <code>[2.0, 3.0, 4.0]</code>, the quantization process will be:
     * </p>
     * <ul>
     *     <li>1.2 < 2.0 -> 0</li>
     *     <li>3.4 > 3.0 -> 1</li>
     *     <li>5.6 > 4.0 -> 1</li>
     * </ul>
     * <p>
     * The quantized vector will be <code>[0, 1, 1]</code>.
     * </p>
     *
     * @param vector     the vector to quantize.
     * @param thresholds the thresholds for quantization, where each element represents the threshold for a corresponding coordinate.
     * @param packedBits the byte array where the quantized bits will be packed.
     */
    void quantizeAndPackBits(final float[] vector, final float[] thresholds, byte[] packedBits) {
        quantizeAndPackBits(vector, new float[][] { thresholds }, 1, packedBits);
    }
}
