/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer quantized byte vectors to off heap memory.
 * The reason this is different from {@link OffHeapBinaryVectorTransfer} is because of allocation and deallocation
 *  of memory on JNI layer. Use this if signed int is needed on JNI layer
 */
public final class OffHeapByteVectorTransfer extends OffHeapVectorTransfer<byte[]> {

    public OffHeapByteVectorTransfer(int bytesPerVector, int totalVectorsToTransfer) {
        super(bytesPerVector, totalVectorsToTransfer);
    }

    @Override
    protected long transfer(List<byte[]> batch, boolean append) throws IOException {
        return JNICommons.storeByteVectorData(
            getVectorAddress(),
            batch.toArray(new byte[][] {}),
            (long) batch.get(0).length * transferLimit,
            append
        );
    }

    @Override
    public void deallocate() {
        JNICommons.freeByteVectorData(getVectorAddress());
    }
}
