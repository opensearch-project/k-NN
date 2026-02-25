/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import java.nio.ByteOrder;

public class FaissFP16Reconstructor extends FaissQuantizedValueReconstructor {
    private final boolean isLittleEndian;

    public FaissFP16Reconstructor(int dimension, int oneVectorElementBits) {
        super(dimension, oneVectorElementBits);
        // FAISS interprets float16 as uint16_t which ultimately interpreted again as uint8_t* then write into a file.
        // That being said, when decoding we need to be careful with system endian, otherwise reconstruction results may have distorted
        // value.
        isLittleEndian = (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void reconstruct(final byte[] quantizedBytes, final float[] floats) {
        if (isLittleEndian) {
            for (int i = 0, j = 0; j < dimension; i += 2, ++j) {
                final short fp16 = (short) ((quantizedBytes[i] & 0xFF) | ((quantizedBytes[i + 1] & 0xFF) << 8));
                floats[j] = Float.float16ToFloat(fp16);
            }
        } else {
            for (int i = 0, j = 0; j < dimension; i += 2, ++j) {
                final short fp16 = (short) (((quantizedBytes[i] & 0xFF) << 8) | (quantizedBytes[i + 1] & 0xFF));
                floats[j] = Float.float16ToFloat(fp16);
            }
        }
    }
}
