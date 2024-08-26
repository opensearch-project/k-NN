/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer quantized binary vectors to off heap memory
 * The reason this is different from {@link OffHeapByteVectorTransfer} is because of allocation and deallocation
 * of memory on JNI layer. Use this if unsigned int is needed on JNI layer
 */
public final class OffHeapBinaryVectorTransfer extends OffHeapVectorTransfer<byte[]> {

    public OffHeapBinaryVectorTransfer(int transferLimit) {
        super(transferLimit);
    }

    @Override
    public void deallocate() {
        JNICommons.freeBinaryVectorData(getVectorAddress());
    }

    @Override
    protected long transfer(List<byte[]> batch, boolean append) throws IOException {
        return JNICommons.storeBinaryVectorData(
            getVectorAddress(),
            batch.toArray(new byte[][] {}),
            (long) batch.get(0).length * transferLimit,
            append
        );
    }
}
