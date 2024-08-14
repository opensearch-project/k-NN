/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer quantized byte vectors to off heap memory.
 * The reason this is different from {@link OffHeapBinaryPreprocessedVectorTransfer} is because of allocation and deallocation
 *  of memory on JNI layer. Use this if signed int is needed on JNI layer
 */
public final class OffHeapBytePreprocessedVectorTransfer<T> extends OffHeapPreprocessedVectorTransfer<T, byte[]> {

    public OffHeapBytePreprocessedVectorTransfer(KNNVectorValues<T> vectorValues, final Long batchSize) throws IOException {
        super(vectorValues, batchSize);
    }

    public OffHeapBytePreprocessedVectorTransfer(KNNVectorValues<T> vectorValues) throws IOException {
        this(vectorValues, null);
    }

    @Override
    protected int computeTransferLimit(byte[] vector) {
        return (int) this.streamingLimit / vector.length;
    }

    @Override
    protected long transfer(List<byte[]> batch, boolean append) throws IOException {
        return JNICommons.storeByteVectorData(getVectorAddress(), batch.toArray(new byte[][] {}), batchSize * batch.get(0).length, append);
    }

    @Override
    public void close() {
        super.close();
        JNICommons.freeByteVectorData(getVectorAddress());
    }
}
