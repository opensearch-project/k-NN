/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import lombok.RequiredArgsConstructor;

/**
 * This factory class creates a reconstructor converting quantized byte values to float array.
 * For example, reconstructing encoded FP16 to FP32.
 */
@RequiredArgsConstructor
public abstract class FaissQuantizedValueReconstructorFactory {
    protected final int dimension;
    protected final int numVectorBytes;

    /**
     * Create reconstructor or return a singleton instance if possible.
     *
     * @return Reconstructor converting quantized bytes to float[]
     */
    public abstract FaissQuantizedValueReconstructor getOrCreate();
}
