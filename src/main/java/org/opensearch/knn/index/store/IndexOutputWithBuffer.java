/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

public class IndexOutputWithBuffer {
    // Underlying `IndexOutput` obtained from Lucene's Directory.
    private IndexOutput indexOutput;
    // Write buffer. Native engine will copy bytes into this buffer.
    // Allocating 64KB here since it show better performance in NMSLIB with the size. (We had slightly improvement in FAISS than having 4KB)
    // NMSLIB writes an adjacent list size first, then followed by serializing the list. Since we usually have more adjacent lists, having
    // 64KB to accumulate bytes as possible to reduce the times of calling `writeBytes`.
    private byte[] buffer = new byte[64 * 1024];

    public IndexOutputWithBuffer(IndexOutput indexOutput) {
        this.indexOutput = indexOutput;
    }

    // This method will be called in JNI layer which precisely knows
    // the amount of bytes need to be written.
    public void writeBytes(int length) {
        try {
            // Delegate Lucene `indexOuptut` to write bytes.
            indexOutput.writeBytes(buffer, 0, length);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "{indexOutput=" + indexOutput + ", len(buffer)=" + buffer.length + "}";
    }
}
