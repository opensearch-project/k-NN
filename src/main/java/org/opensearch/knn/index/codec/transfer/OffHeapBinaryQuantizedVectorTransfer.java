/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.List;

/**
 * Transfer quantized binary vectors to off heap memory
 * The reason this is different from {@link OffHeapByteQuantizedVectorTransfer} is because of allocation and deallocation
 * of memory on JNI layer. Use this if unsigned int is needed on JNI layer
 */
public final class OffHeapBinaryQuantizedVectorTransfer<T> extends OffHeapQuantizedVectorTransfer<T, byte[]> {

    public OffHeapBinaryQuantizedVectorTransfer(KNNVectorValues<T> vectorValues, Long batchSize) {
        super(vectorValues, batchSize, (vector, state) -> (byte[]) vector, StringUtils.EMPTY, DEFAULT_COMPRESSION_FACTOR);
    }

    public OffHeapBinaryQuantizedVectorTransfer(KNNVectorValues<T> vectorValues) {
        this(vectorValues, null);
    }

    @Override
    public void close() {
        super.close();
        // TODO: deallocate the memory location
    }

    @Override
    protected long transfer(List<byte[]> vectorsToTransfer, boolean append) throws IOException {
        // TODO: call to JNIService to transfer vector
        return 0L;
    }
}
