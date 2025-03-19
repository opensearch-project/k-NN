/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

/**
 * This reconstructs a float from bytes encoded in a signed 8-bit, byte-compressed format.
 *
 * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.cpp#L856">FAISS Decode</a>
 */
public class Faiss8BitsDirectSignedReconstructorFactory extends FaissQuantizedValueReconstructorFactory {
    private Faiss8BitsDirectSignedReconstructor reconstructor;

    public Faiss8BitsDirectSignedReconstructorFactory(int dimension, int numVectorBytes) {
        super(dimension, numVectorBytes);
        reconstructor = new Faiss8BitsDirectSignedReconstructor(dimension, numVectorBytes);
    }

    @Override
    public FaissQuantizedValueReconstructor getOrCreate() {
        return reconstructor;
    }

    private static class Faiss8BitsDirectSignedReconstructor extends FaissQuantizedValueReconstructor {
        public Faiss8BitsDirectSignedReconstructor(int dimension, int numVectorBytes) {
            super(dimension, numVectorBytes);
        }

        @Override
        public void reconstruct(byte[] bytes, float[] floats) {
            for (int i = 0; i < dimension; ++i) {
                // bytes[i] should be interpreted as uint8_t.
                // Hence, we can't just cast it to int.
                // Ex: uint8_t x = 200 -> -56 as byte in Java.
                //     With int casting, then it will become -56 rather than 200.
                floats[i] = (bytes[i] & 0xFF) - 128;
            }
        }
    }
}
