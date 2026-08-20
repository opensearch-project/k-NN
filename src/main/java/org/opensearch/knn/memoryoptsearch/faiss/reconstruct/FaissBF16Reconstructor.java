/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import java.nio.ByteOrder;

public class FaissBF16Reconstructor extends FaissQuantizedValueReconstructor {
    private final boolean isLittleEndian;

    public FaissBF16Reconstructor(int dimension, int oneVectorElementBits) {
        // Similar to FP16, FAISS stores BF16 values as uint16_t which is ultimately interpreted as uint8_t*
        // written into a file. We need to handle endianness during decoding.
        this(dimension, oneVectorElementBits, ByteOrder.nativeOrder());
    }

    FaissBF16Reconstructor(int dimension, int oneVectorElementBits, ByteOrder byteOrder) {
        super(dimension, oneVectorElementBits);
        this.isLittleEndian = (byteOrder == ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void reconstruct(final byte[] quantizedBytes, final float[] floats) {
        // BF16 is the upper 16 bits of a float32.
        // To convert, we shift the BF16 bits left by 16 positions and interpret as float32.
        if (isLittleEndian) {
            for (int i = 0, j = 0; j < dimension; i += 2, ++j) {
                final int bf16Bits = (quantizedBytes[i] & 0xFF) | ((quantizedBytes[i + 1] & 0xFF) << 8);
                floats[j] = Float.intBitsToFloat(bf16Bits << 16);
            }
        } else {
            for (int i = 0, j = 0; j < dimension; i += 2, ++j) {
                final int bf16Bits = ((quantizedBytes[i] & 0xFF) << 8) | (quantizedBytes[i + 1] & 0xFF);
                floats[j] = Float.intBitsToFloat(bf16Bits << 16);
            }
        }
    }
}
