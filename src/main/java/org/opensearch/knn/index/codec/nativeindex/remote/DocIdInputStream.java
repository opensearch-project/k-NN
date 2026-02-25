/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * {@link InputStream} implementation of doc ids backed by {@link KNNVectorValues} rather than any file. Intended for use by {@link RemoteIndexBuildStrategy}
 */
@Log4j2
class DocIdInputStream extends InputStream {
    private final KNNVectorValues<?> knnVectorValues;
    // Doc ids are 4 byte integers, byte read() only returns a single byte, so we will need to track the byte position within a doc id.
    // For simplicity, and to maintain the byte ordering, we use a buffer with size of 1 int.
    private ByteBuffer currentBuffer;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Use to represent the doc ids of a {@link KNNVectorValues} as an {@link InputStream}. Expected to be used only with {@link org.opensearch.common.blobstore.BlobContainer#writeBlob}.
     * @param knnVectorValues
     * @throws IOException
     * @see VectorValuesInputStream
     */
    public DocIdInputStream(KNNVectorValues<?> knnVectorValues) throws IOException {
        this.currentBuffer = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.knnVectorValues = knnVectorValues;
        initializeVectorValues(this.knnVectorValues);
        reloadBuffer();
    }

    @Override
    public int read() throws IOException {
        checkClosed();
        if (currentBuffer == null) {
            return -1;
        }

        if (!currentBuffer.hasRemaining()) {
            advanceAndReloadBuffer();
            if (currentBuffer == null) {
                return -1;
            }
        }

        // Unsigned byte conversion is not technically needed as we are using a ByteBuffer, however we perform this operation still just in
        // case.
        return currentBuffer.get() & 0xFF;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        checkClosed();
        if (currentBuffer == null) {
            return -1;
        }

        int available = currentBuffer.remaining();
        if (available <= 0) {
            advanceAndReloadBuffer();
            if (currentBuffer == null) {
                return -1;
            }
            available = currentBuffer.remaining();
        }

        int bytesToRead = Math.min(available, len);
        currentBuffer.get(b, off, bytesToRead);
        return bytesToRead;
    }

    /**
     * Marks this stream as closed
     * @throws IOException
     */
    @Override
    public void close() throws IOException {
        super.close();
        currentBuffer = null;
        closed.set(true);
    }

    private void checkClosed() throws IOException {
        if (closed.get()) {
            throw new IOException("Stream closed");
        }
    }

    /**
     * Advances to the next doc, and then refills the buffer with the new doc.
     * @throws IOException
     */
    private void advanceAndReloadBuffer() throws IOException {
        int docId = knnVectorValues.nextDoc();
        if (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS) {
            reloadBuffer();
        } else {
            // Reset buffer to null to indicate that there are no more docs to be read
            currentBuffer = null;
        }
    }

    /**
     * Reload {@link currentBuffer} with the current doc id that {@link knnVectorValues} is pointing to
     * @throws IOException
     */
    private void reloadBuffer() throws IOException {
        currentBuffer.clear();
        currentBuffer.putInt(knnVectorValues.docId());
        currentBuffer.position(0);
    }
}
