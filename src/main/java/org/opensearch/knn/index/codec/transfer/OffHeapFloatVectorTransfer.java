/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer float vectors to off heap memory.
 */
public final class OffHeapFloatVectorTransfer extends OffHeapVectorTransfer<float[]> {

    public OffHeapFloatVectorTransfer(int bytesPerVector, int totalVectorsToTransfer) {
        super(bytesPerVector, totalVectorsToTransfer);
    }

    @Override
    protected long transfer(final List<float[]> vectorsToTransfer, boolean append) throws IOException {
        return JNICommons.storeVectorData(
            getVectorAddress(),
            vectorsToTransfer.toArray(new float[][] {}),
            (long) vectorsToTransfer.get(0).length * this.transferLimit,
            append
        );
    }

    @Override
    public void deallocate() {
        JNICommons.freeVectorData(getVectorAddress());
    }
}
