/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;

/**
 * This reconstructs quantized bytes to float array.
 * For example, reconstructing encoded FP16 to FP32.
 */
@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
public abstract class FaissQuantizedValueReconstructor {
    protected final int dimension;
    protected final int oneVectorElementBits;

    /**
     * Reconstruct float32 array from quantized bytes.
     * @param quantizedBytes Quantized byte stream.
     * @param floats Destination float array buffer to reconstruct.
     */
    public void reconstruct(byte[] quantizedBytes, float[] floats) {
        throw new UnsupportedOperationException();
    }

    /**
     * Reconstruct quantized bytes to byte[].
     * This reconstruct method may not be applied for all quantization type.
     * For example, FP16, which encode Flat32 into two bytes, cannot use this method but must use
     * 'void reconstruct(byte[] quantizedBytes, float[] floats)'
     * <p>
     * Length of `bytes` must be greater than or equal to the length of `quantizedBytes`, must not be null.
     * It is allowed to pass the identical byte[] as parameters.
     * <p>
     * Ex:
     * byte[] quantizedBytes = read(...);
     * reconstruct(quantizedBytes, quantizedBytes);  -> this decode quantized byte then put it back to the passed byte array.
     *
     * @param quantizedBytes Quantized byte stream.
     * @param bytes Destination float array buffer to reconstruct.
     */
    public void reconstruct(byte[] quantizedBytes, byte[] bytes) {
        throw new UnsupportedOperationException();
    }
}
