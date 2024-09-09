/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.Getter;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 * The class is intended to transfer {@link KNNVectorValues} to off heap memory.
 * </p>
 * <p>
 * The class is not thread safe.
 * </p>
 *
 * @param <T>  byte[] or float[]
 */
public abstract class OffHeapVectorTransfer<T> implements Closeable {

    @Getter
    private long vectorAddress;
    @Getter
    protected final int transferLimit;

    private List<T> vectorsToTransfer;

    public OffHeapVectorTransfer(int bytesPerVector, int totalVectorsToTransfer) {
        this.transferLimit = computeTransferLimit(bytesPerVector, totalVectorsToTransfer);
        this.vectorsToTransfer = new ArrayList<>(this.transferLimit);
        this.vectorAddress = 0;
    }

    private int computeTransferLimit(int bytesPerVector, int totalVectorsToTransfer) {
        int limit = (int) Math.max(1, KNNSettings.getVectorStreamingMemoryLimit().getBytes() / bytesPerVector);
        return Math.min(limit, totalVectorsToTransfer);
    }

    /**
     * Transfer vectors to off-heap
     * @param vector float[] or byte[]
     * @param append This indicates whether to append or rewrite the off-heap buffer
     * @return true of the vectors were transferred, false if not
     * @throws IOException
     */
    public boolean transfer(T vector, boolean append) throws IOException {
        vectorsToTransfer.add(vector);
        if (vectorsToTransfer.size() == this.transferLimit) {
            vectorAddress = transfer(vectorsToTransfer, append);
            vectorsToTransfer.clear();
            return true;
        }
        return false;
    }

    /**
     * Empties the {@link #vectorsToTransfer} if its not empty. Intended to be used before
     * closing the transfer
     *
     * @param append This indicates whether to append or rewrite the off-heap buffer
     * @return true of the vectors were transferred, false if not
     * @throws IOException
     */
    public boolean flush(boolean append) throws IOException {
        // flush before closing
        if (!vectorsToTransfer.isEmpty()) {
            vectorAddress = transfer(vectorsToTransfer, append);
            vectorsToTransfer.clear();
            return true;
        }
        return false;
    }

    @Override
    public void close() {
        // Remove this if condition once create and write index is separated for nmslib
        if (vectorAddress != 0) {
            deallocate();
        }
        reset();
    }

    /**
     * Resets address and vectortoTransfer
     *
     * DO NOT USE this in the middle of the transfer, The behavior is undefined
     *
     * TODO: Make it package private once create and write index is separated for nmslib
     */
    public void reset() {
        vectorAddress = 0;
        vectorsToTransfer = null;
    }

    protected abstract void deallocate();

    protected abstract long transfer(final List<T> vectorsToTransfer, boolean append) throws IOException;
}
