/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * This class contains a Lucene's IndexInput with a reader buffer.
 * A Java reference of this class will be passed to native engines, then 'copyBytes' method will be
 * called by native engine via JNI API.
 * Therefore, this class servers as a read layer in native engines to read the bytes it wants.
 */
public class IndexInputWithBuffer {
    private IndexInput indexInput;
    // 4K buffer.
    private byte[] buffer = new byte[4 * 1024];

    public IndexInputWithBuffer(@NonNull IndexInput indexInput) {
        this.indexInput = indexInput;
    }

    /**
     * This method will be invoked in native engines via JNI API.
     * Then it will call IndexInput to read required bytes then copy them into a read buffer.
     *
     * @param nbytes Desired number of bytes to be read.
     * @return The number of read bytes in a buffer.
     * @throws IOException
     */
    private int copyBytes(long nbytes) throws IOException {
        final int readBytes = (int) Math.min(nbytes, buffer.length);
        indexInput.readBytes(buffer, 0, readBytes);
        return readBytes;
    }

    @Override
    public String toString() {
        return "{indexInput=" + indexInput + ", len(buffer)=" + buffer.length + "}";
    }
}
