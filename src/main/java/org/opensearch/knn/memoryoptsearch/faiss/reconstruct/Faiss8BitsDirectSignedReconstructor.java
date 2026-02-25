/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

public class Faiss8BitsDirectSignedReconstructor extends FaissQuantizedValueReconstructor {
    public Faiss8BitsDirectSignedReconstructor(int dimension, int oneVectorElementBits) {
        super(dimension, oneVectorElementBits);
    }

    @Override
    public void reconstruct(byte[] quantizedBytes, byte[] bytes) {
        for (int i = 0; i < dimension; ++i) {
            // bytes[i] should be interpreted as uint8_t.
            // Hence, we can't just cast it to int.
            // Ex: uint8_t x = 200 -> -56 as byte in Java.
            // With int casting, then it will become -56 rather than 200, which is incorrect conversion.
            bytes[i] = (byte) ((quantizedBytes[i] & 0xFF) - 128);
        }
    }
}
