/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.Getter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 * The class is intended to transfer {@link KNNVectorValues} to off heap memory.
 * It also provides and ability to quantize the vector before it is transferred to offHeap memory.
 * The ability to quantize is added as to not iterate KNN {@link KNNVectorValues} multiple times.
 * </p>
 *
 * <p>
 * The class is not thread safe.
 * </p>
 *
 * @param <T>  byte[] or float[]
 */
public abstract class OffHeapVectorTransfer<T> implements Closeable {

    @Getter
    private long vectorAddress;
    protected final int transferLimit;

    private final List<T> vectorsToTransfer;

    public OffHeapVectorTransfer(final int transferLimit) {
        this.transferLimit = transferLimit;
        this.vectorsToTransfer = new ArrayList<>(transferLimit);
        this.vectorAddress = 0;
    }

    public boolean transfer(T vector, boolean append) throws IOException {
        vectorsToTransfer.add(vector);
        if (vectorsToTransfer.size() == this.transferLimit) {
            vectorAddress = transfer(vectorsToTransfer, append);
            vectorsToTransfer.clear();
            return true;
        }
        return false;
    }

    public boolean flush(boolean append) throws IOException {
        // flush before closing
        if (!vectorsToTransfer.isEmpty()) {
            vectorAddress = transfer(vectorsToTransfer, append);
            return true;
        }
        return false;
    }

    public void close() {
        vectorAddress = 0;
    }

    protected abstract long transfer(final List<T> vectorsToTransfer, boolean append) throws IOException;
}
