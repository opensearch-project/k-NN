/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import java.nio.ByteOrder;

/**
 * This reconstructs a float from FP16 encoded in short (e.g. uint16_t), two bytes compressed format.
 *
 * FYI : <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/utils/fp16-inl.h#L89">FAISS Decoding</a>
 */
public class FaissFP16ReconstructorFactory extends FaissQuantizedValueReconstructorFactory {
    private final FaissFP16Reconstructor reconstructor;

    public FaissFP16ReconstructorFactory(int dimension, int numVectorBytes) {
        super(dimension, numVectorBytes);
        reconstructor = new FaissFP16Reconstructor(dimension, numVectorBytes);
    }

    @Override
    public FaissQuantizedValueReconstructor getOrCreate() {
        return reconstructor;
    }

    private static class FaissFP16Reconstructor extends FaissQuantizedValueReconstructor {
        private final boolean isLittleEndian;

        public FaissFP16Reconstructor(int dimension, int numVectorBytes) {
            super(dimension, numVectorBytes);
            // FAISS interprets float16 as uint16_t which ultimately interpreted again as uint8_t* then write into a file.
            // That being said, when decoding we need to be careful with system endian, otherwise reconstruction results may have distorted
            // value.
            isLittleEndian = (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void reconstruct(final byte[] bytes, final float[] floats) {
            if (isLittleEndian) {
                for (int i = 0, j = 0; i < numVectorBytes; i += 2, ++j) {
                    final short fp16 = (short) ((bytes[i] & 0xFF) | ((bytes[i + 1] & 0xFF) << 8));
                    floats[j] = Float.float16ToFloat(fp16);
                }
            } else {
                for (int i = 0, j = 0; i < numVectorBytes; i += 2, ++j) {
                    final short fp16 = (short) (((bytes[i] & 0xFF) << 8) | (bytes[i + 1] & 0xFF));
                    floats[j] = Float.float16ToFloat(fp16);
                }
            }
        }
    }
}
