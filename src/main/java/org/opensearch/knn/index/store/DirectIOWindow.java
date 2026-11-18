/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import java.nio.ByteBuffer;

/**
 * The aligned direct buffer a {@link DirectIOVectorIndexInput} reads into. The random readers on a thread share one, since
 * Lucene never closes the slices and clones it makes per query, and a window serves bytes back only to the input that read them.
 */
final class DirectIOWindow {

    private static final ThreadLocal<DirectIOWindow> THREAD_WINDOWS = ThreadLocal.withInitial(DirectIOWindow::new);

    private ByteBuffer buffer;
    private int blockSize;
    private long owner = -1;
    private long start = -1;
    private int valid;

    static DirectIOWindow ofThread() {
        return THREAD_WINDOWS.get();
    }

    // Allocates the buffer on first use, and again when a reader needs it larger or aligned to another block size.
    ByteBuffer buffer(final int bytes, final int blockSize) {
        if (buffer == null || buffer.capacity() < bytes || this.blockSize != blockSize) {
            buffer = DirectIOFile.allocateAligned(bytes, blockSize);
            this.blockSize = blockSize;
            clear();
        }
        return buffer;
    }

    // The index of the byte at an absolute offset, or -1 unless this reader's last fill read it.
    int indexOf(final long reader, final long absolute) {
        if (owner != reader || absolute < start || absolute >= start + valid) {
            return -1;
        }
        return (int) (absolute - start);
    }

    int valid() {
        return valid;
    }

    void filled(final long reader, final long alignedStart, final int bytes) {
        owner = reader;
        start = alignedStart;
        valid = bytes;
    }

    void clear() {
        owner = -1;
        start = -1;
        valid = 0;
    }
}
