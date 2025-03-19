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
    protected final int numVectorBytes;

    /**
     * Reconstruct float32 array from quantized bytes.
     * @param bytes Quantized byte stream.
     * @param floats Destination float array buffer to reconstruct.
     */
    public abstract void reconstruct(byte[] bytes, float[] floats);
}
