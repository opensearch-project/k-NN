/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.exception.TerminalIOException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.NoSuchFileException;
import java.util.Arrays;

/**
 * Wrapper around {@link IndexOutput} to perform writes in a buffered manner. This class is created per flush/merge, and may be used twice if
 * {@link org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy} needs to fall back to a different build strategy.
 */
@Log4j2
public class IndexOutputWithBuffer implements AutoCloseable {
    // Underlying `IndexOutput` obtained from Lucene's Directory.
    private IndexOutput indexOutput;
    // Write buffer. Native engine will copy bytes into this buffer.
    // Allocating 64KB here since it show better performance in NMSLIB with the size. (We had slightly improvement in FAISS than having 4KB)
    // NMSLIB writes an adjacent list size first, then followed by serializing the list. Since we usually have more adjacent lists, having
    // 64KB to accumulate bytes as possible to reduce the times of calling `writeBytes`.
    private static final int CHUNK_SIZE = 64 * 1024;
    private final byte[] buffer;
    @Getter
    private long bytesWritten;

    public IndexOutputWithBuffer(IndexOutput indexOutput) {
        this.indexOutput = indexOutput;
        this.buffer = new byte[CHUNK_SIZE];
        this.bytesWritten = 0;
    }

    // This method will be called in JNI layer which precisely knows
    // the amount of bytes need to be written.
    public void writeBytes(int length) {
        try {
            // Delegate Lucene `indexOuptut` to write bytes.
            indexOutput.writeBytes(buffer, 0, length);
            bytesWritten += length;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Writes to the {@link IndexOutput} by buffering bytes into a new buffer of custom size.
     *
     * @param inputStream       The stream from which we are reading bytes to write
     * @throws IOException
     * @see IndexOutputWithBuffer#writeFromStreamWithBuffer(InputStream, byte[])
     */
    public void writeFromStreamWithBuffer(InputStream inputStream, int size) throws IOException {
        byte[] streamBuffer = new byte[size];
        writeFromStreamWithBuffer(inputStream, streamBuffer);
    }

    /**
     * Writes to the {@link IndexOutput} by buffering bytes with @param outputBuffer. This method allows
     * {@link org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy} to provide a separate, larger buffer as that buffer is for buffering
     * bytes downloaded from the repository, so it may be more performant to use a larger buffer.
     * We do not change the size of the existing buffer in case a fallback to the existing build strategy is needed.
     *
     * @param inputStream       The stream from which we are reading bytes to write
     * @param outputBuffer      The buffer used to buffer bytes
     * @throws IOException
     * @see IndexOutputWithBuffer#writeFromStreamWithBuffer(InputStream, int)
     */
    private void writeFromStreamWithBuffer(InputStream inputStream, byte[] outputBuffer) throws IOException {
        int bytesRead = 0;
        // InputStream uses -1 indicates there are no more bytes to be read
        while (bytesRead != -1) {
            // Try to read enough bytes to fill the entire buffer. The actual amount read may be less.
            bytesRead = inputStream.read(outputBuffer, 0, outputBuffer.length);
            assert bytesRead <= outputBuffer.length;
            // However many bytes we read, write it to the IndexOutput if != -1
            if (bytesRead != -1) {
                try {
                    indexOutput.writeBytes(outputBuffer, 0, bytesRead);
                    bytesWritten += bytesRead;
                } catch (IOException e) {
                    throw new TerminalIOException("Failed to write to indexOutput", e);
                }
            }
        }
    }

    /**
     * Returns the name of the underlying output file.
     */
    public String getName() {
        return indexOutput.getName();
    }

    /**
     * Resets this buffer by closing the current {@link IndexOutput}, deleting the partially written
     * file, and opening a fresh output with the same file name. This is used when a remote index
     * build partially writes data and then fails — the fallback strategy needs a clean output.
     *
     * @param directory the directory to create the new output in
     * @param context   the IO context for creating the new output
     * @throws IOException if closing, deleting, or creating the output fails
     */
    public void reset(final Directory directory, final IOContext context) throws IOException {
        final String fileName = indexOutput.getName();
        indexOutput.close();
        try {
            directory.deleteFile(fileName);
        } catch (NoSuchFileException e) {
            // File may not exist if nothing was written — safe to ignore
            log.debug("File not found: {}, while trying to delete the file", fileName);
        }
        this.indexOutput = directory.createOutput(fileName, context);
        bytesWritten = 0;
        Arrays.fill(buffer, (byte) 0);
    }

    /**
     * Writes the Lucene codec footer to the underlying {@link IndexOutput}.
     */
    public void writeFooter() throws IOException {
        CodecUtil.writeFooter(indexOutput);
    }

    @Override
    public void close() throws IOException {
        indexOutput.close();
    }

    @Override
    public String toString() {
        return "{indexOutput=" + indexOutput + ", len(buffer)=" + buffer.length + "}";
    }
}
