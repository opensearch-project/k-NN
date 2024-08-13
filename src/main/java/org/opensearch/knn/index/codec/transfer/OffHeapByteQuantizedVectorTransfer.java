/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer quantized byte vectors to off heap memory.
 * The reason this is different from {@link OffHeapBinaryQuantizedVectorTransfer} is because of allocation and deallocation
 *  of memory on JNI layer. Use this if signed int is needed on JNI layer
 */
public final class OffHeapByteQuantizedVectorTransfer<T> extends OffHeapQuantizedVectorTransfer<T, byte[]> {

    public OffHeapByteQuantizedVectorTransfer(KNNVectorValues<T> vectorValues, final Long batchSize) throws IOException {
        super(vectorValues, batchSize, (vector, state) -> (byte[]) vector, StringUtils.EMPTY, DEFAULT_COMPRESSION_FACTOR);
    }

    public OffHeapByteQuantizedVectorTransfer(KNNVectorValues<T> vectorValues) throws IOException {
        this(vectorValues, null);
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
