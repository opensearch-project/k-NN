/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.QuantizedKNNBinaryVectorValues;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.VectorDataType.BINARY;
import static org.opensearch.knn.index.VectorDataType.BYTE;
import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * {@link InputStream} implementation backed by {@link KNNVectorValues} rather than any file. Intended for use by {@link RemoteIndexBuildStrategy}
 */
@Log4j2
class VectorValuesInputStream extends InputStream {

    private final KNNVectorValues<?> knnVectorValues;
    // It is difficult to avoid using a buffer in this class as we need to be able to convert from float[] to byte[]. this buffer
    // will be filled 1 vector at a time.
    private ByteBuffer currentBuffer;
    private final int bytesPerVector;
    private long bytesRemaining;
    private final VectorDataType vectorDataType;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * Used to represent a part of a {@link KNNVectorValues} as an {@link InputStream}. Expected to be used with
     * {@link org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer#asyncBlobUpload}. The repository will interact with this class only through the constructor and the read methods.
     * <p>
     *     Note: For S3 (but generically too), the retryable input stream is backed by a buffer with the same size as the stream, so all bytes are loaded onto heap
     *     at once (16mb chunks by default) when a given {@link VectorValuesInputStream} is being processed.
     * </p>
     * <p>
     *     Note: The S3 implementation will only request up to {@param size} bytes from this InputStream. However, that is implementation specific and may not be
     *     true for all implementations, so we do our own size enforcement here as well.
     * </p>
     *
     * @param knnVectorValues
     * @param vectorDataType
     * @param startPosition
     * @param size
     * @throws IOException
     * @see DocIdInputStream
     */
    public VectorValuesInputStream(KNNVectorValues<?> knnVectorValues, VectorDataType vectorDataType, long startPosition, long size)
        throws IOException {
        this.bytesRemaining = size;
        this.knnVectorValues = knnVectorValues;
        this.vectorDataType = vectorDataType;
        initializeVectorValues(this.knnVectorValues);
        this.bytesPerVector = this.knnVectorValues.bytesPerVector();
        // We use currentBuffer == null to indicate that there are no more vectors to be read
        this.currentBuffer = ByteBuffer.allocate(bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        // Position the InputStream at the specific byte within the specific vector that startPosition references
        setPosition(startPosition);
    }

    /**
     * Used to represent the entire {@link KNNVectorValues} as a single {@link InputStream}. Expected to be used with
     * {@link org.opensearch.common.blobstore.BlobContainer#writeBlob}
     *
     * @param knnVectorValues
     * @param vectorDataType
     * @throws IOException
     * @see DocIdInputStream
     */
    public VectorValuesInputStream(KNNVectorValues<?> knnVectorValues, VectorDataType vectorDataType) throws IOException {
        this(knnVectorValues, vectorDataType, 0, Long.MAX_VALUE);
    }

    @Override
    public int read() throws IOException {
        checkClosed();
        if (bytesRemaining <= 0 || currentBuffer == null) {
            return -1;
        }

        if (!currentBuffer.hasRemaining()) {
            advanceAndReloadBuffer();
            if (currentBuffer == null) {
                return -1;
            }
        }

        bytesRemaining--;
        // Unsigned byte conversion is not technically needed as we are using a ByteBuffer, however we perform this operation still just in
        // case.
        return currentBuffer.get() & 0xFF;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        checkClosed();
        if (bytesRemaining <= 0 || currentBuffer == null) {
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
        int bytesActuallyRead = (int) Math.min(bytesRemaining, bytesToRead);
        currentBuffer.get(b, off, bytesActuallyRead);
        bytesRemaining -= bytesActuallyRead;
        return bytesActuallyRead;
    }

    /**
     * This class does not support skipping. Instead, use {@link VectorValuesInputStream#setPosition}.
     *
     * @param n   the number of bytes to be skipped.
     * @return
     * @throws IOException
     */
    @Override
    public long skip(long n) throws IOException {
        checkClosed();
        throw new UnsupportedOperationException("VectorValuesInputStream does not support skip");
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
     * Advances n bytes forward in the knnVectorValues.
     * Note: {@link KNNVectorValues#advance} is not supported when we are merging segments, so we do not use it here.
     * Note: {@link KNNVectorValues#nextDoc} is relatively efficient, but {@link KNNVectorValues#getVector} may
     * perform a disk read, so we avoid using {@link VectorValuesInputStream#reloadBuffer()} here.
     *
     * @param n
     * @return
     * @throws IOException
     */
    private void setPosition(long n) throws IOException {
        if (currentBuffer.position() != 0) {
            throw new UnsupportedOperationException("setPosition is only supported from the start of a vector");
        }

        long bytesSkipped = 0;
        int vectorsToSkip = (int) (n / bytesPerVector);
        log.debug("Skipping {} bytes, {} vectors", n, vectorsToSkip);
        int docId = knnVectorValues.docId();
        while (docId != -1 && docId != DocIdSetIterator.NO_MORE_DOCS && vectorsToSkip > 0) {
            docId = knnVectorValues.nextDoc();
            bytesSkipped += bytesPerVector;
            vectorsToSkip--;
        }

        // After skipping the correct number of vectors, fill the buffer with the current vector
        reloadBuffer();

        // Advance to the correct position within the current vector
        long remainingBytes = n - bytesSkipped;
        if (remainingBytes > 0) {
            currentBuffer.position((int) remainingBytes);
        }
    }

    /**
     * Reload {@link currentBuffer} with the current vector that {@link knnVectorValues} is pointing to
     * @throws IOException
     */
    private void reloadBuffer() throws IOException {
        currentBuffer.clear();
        if (vectorDataType == FLOAT) {
            float[] floatVector = ((KNNFloatVectorValues) knnVectorValues).getVector();
            currentBuffer.asFloatBuffer().put(floatVector);
        } else if (vectorDataType == BYTE) {
            byte[] byteVector = ((KNNByteVectorValues) knnVectorValues).getVector();
            currentBuffer.put(byteVector);
        } else if (vectorDataType == BINARY) {
            final byte[] binaryVector;
            if (knnVectorValues instanceof QuantizedKNNBinaryVectorValues quantizedKNNBinaryVectorValues) {
                // Original vector is non-binary, and we applied quantization on them to binary vectors.
                binaryVector = quantizedKNNBinaryVectorValues.getVector();
            } else {
                // Original vector is already binary vectors, hence there's no quantization status
                binaryVector = ((KNNBinaryVectorValues) knnVectorValues).getVector();
            }
            currentBuffer.put(binaryVector);
        } else {
            throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
        // Reset buffer to head for future reads
        currentBuffer.position(0);
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
}
