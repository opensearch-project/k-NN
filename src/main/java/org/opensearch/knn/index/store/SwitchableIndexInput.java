/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.opensearch.common.util.io.IOUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

/**
 * The input {@link KNNDirectIODirectory} hands out for a flat vector data file that could be read with
 * {@code O_DIRECT}. It forwards every method to the input the original directory opened, and at each slice and each
 * clone it hands out either that input's own slice or clone or a {@link DirectIOVectorIndexInput}, by the index
 * setting, in both directions.
 *
 * <p>While the index is off every slice and clone is the original input's own object, so the file reads exactly as it
 * does without this delegate and nothing is opened. The first slice, clone or efficiency question taken while the
 * index is on opens the {@code O_DIRECT} channel, runs the content check and keeps the answer for the rest of this
 * input's life.
 *
 * <p>Its own reads always go to the original input: what reads a file input directly rather than through a slice or a
 * clone is the codec header and the footer checksum at reader open.
 */
public final class SwitchableIndexInput extends IndexInput {

    private final KNNDirectIODirectory directory;
    private final IndexInput originalInput;
    private final Path path;
    private final long fileLength;
    private final boolean sequentialOpen;
    private final DirectIOFile.ChannelFactory channelFactory;

    private boolean initialized;
    private DirectIOFile directIOFile;
    private volatile boolean closed;

    SwitchableIndexInput(
        @NonNull final KNNDirectIODirectory directory,
        @NonNull final String name,
        @NonNull final Path path,
        @NonNull final IndexInput originalInput,
        final long fileLength,
        final boolean sequentialOpen,
        @NonNull final DirectIOFile.ChannelFactory channelFactory
    ) {
        super("SwitchableIndexInput(" + name + ")");
        this.directory = directory;
        this.path = path;
        this.originalInput = originalInput;
        this.fileLength = fileLength;
        this.sequentialOpen = sequentialOpen;
        this.channelFactory = channelFactory;
    }

    // Opens and checks this file's O_DIRECT channel, once. Called from the first slice or clone taken while the index
    // is on.
    private synchronized boolean initializeDirectIO() {
        if (closed) {
            // An input that is closing or closed opens nothing: the channel would have no owner left to close it.
            return false;
        }
        if (initialized == false) {
            initialized = true;
            directIOFile = DirectIOFile.open(path, fileLength, originalInput, channelFactory);
        }
        return directIOFile != null;
    }

    // The file a new slice or clone reads through, or null if the index is off, this file declined or this input is
    // closed. Initializing and closing hold the same monitor, so a channel is opened at most once and is never left
    // without an owner. A caller that slices or clones concurrently with a close can be handed a channel this input is
    // closing, which Lucene's contract rules out anyway.
    private DirectIOFile directIOFileToUse() {
        // The switch is read outside the monitor: it is a cluster state lookup and no caller should wait on another
        // caller's initialization to learn that the index is off.
        if (directory.useDirectIO() == false) {
            return null;
        }
        synchronized (this) {
            return initializeDirectIO() ? directIOFile : null;
        }
    }

    // Either the original input's own slice or clone, which is what a file that is off or declined hands out, or a
    // Direct I/O reader over it.
    private IndexInput directIOOrOriginal(
        final DirectIOFile file,
        final String sliceDescription,
        final IndexInput original,
        final long offset,
        final long length,
        final boolean sequential,
        final long filePointer
    ) {
        if (file == null) {
            return original;
        }
        return new DirectIOVectorIndexInput(
            getFullSliceDescription(sliceDescription),
            file,
            original,
            offset,
            length,
            KNNDirectIODirectory.bufferSizeFor(sequential, file.blockSize()),
            filePointer
        );
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
        ensureOpen();
        final DirectIOFile file = directIOFileToUse();
        return directIOOrOriginal(
            file,
            sliceDescription,
            originalInput.slice(sliceDescription, offset, length),
            offset,
            length,
            sequentialOpen,
            0L
        );
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length, final IOContext context)
        throws IOException {
        ensureOpen();
        final DirectIOFile file = directIOFileToUse();
        // The original input's own overload, which on an mmap input applies the context's read advice to the slice.
        // Composing the three-argument slice with updateIOContext would not: mmap does not implement it.
        return directIOOrOriginal(
            file,
            sliceDescription,
            originalInput.slice(sliceDescription, offset, length, context),
            offset,
            length,
            sequentialOpen || KNNDirectIODirectory.isSequential(context),
            0L
        );
    }

    @Override
    public IndexInput clone() {
        ensureOpen();
        final DirectIOFile file = directIOFileToUse();
        final IndexInput originalClone = originalInput.clone();
        return directIOOrOriginal(file, "clone", originalClone, 0L, fileLength, sequentialOpen, originalClone.getFilePointer());
    }

    @Override
    public RandomAccessInput randomAccessSlice(final long offset, final long length) throws IOException {
        ensureOpen();
        // A sparse field's ord-to-doc table is read through this, one lookup at a time, which on an aligned blocking
        // input would be a device read per lookup. It is small next to the vectors, so it stays on the original input.
        return originalInput.randomAccessSlice(offset, length);
    }

    // A file that is not read with O_DIRECT answers exactly what the original input answers, or Lucene would open the
    // file a second and a third time for a merge and a sequential search.
    @Override
    public boolean isEfficientFor(final IOContext context) {
        ensureOpen();
        if (directIOFileToUse() == null) {
            return originalInput.isEfficientFor(context);
        }
        return sequentialOpen || KNNDirectIODirectory.isSequential(context) == false;
    }

    @Override
    public byte readByte() throws IOException {
        ensureOpen();
        return originalInput.readByte();
    }

    @Override
    public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
        ensureOpen();
        originalInput.readBytes(b, offset, len);
    }

    @Override
    public void readBytes(final byte[] b, final int offset, final int len, final boolean useBuffer) throws IOException {
        ensureOpen();
        originalInput.readBytes(b, offset, len, useBuffer);
    }

    @Override
    public void readInts(final int[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        originalInput.readInts(dst, offset, len);
    }

    @Override
    public void readLongs(final long[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        originalInput.readLongs(dst, offset, len);
    }

    @Override
    public void readFloats(final float[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        originalInput.readFloats(dst, offset, len);
    }

    @Override
    public short readShort() throws IOException {
        ensureOpen();
        return originalInput.readShort();
    }

    @Override
    public int readInt() throws IOException {
        ensureOpen();
        return originalInput.readInt();
    }

    @Override
    public int readVInt() throws IOException {
        ensureOpen();
        return originalInput.readVInt();
    }

    @Override
    public long readLong() throws IOException {
        ensureOpen();
        return originalInput.readLong();
    }

    @Override
    public long readVLong() throws IOException {
        ensureOpen();
        return originalInput.readVLong();
    }

    @Override
    public long getFilePointer() {
        ensureOpen();
        return originalInput.getFilePointer();
    }

    @Override
    public void seek(final long pos) throws IOException {
        ensureOpen();
        originalInput.seek(pos);
    }

    @Override
    public void skipBytes(final long numBytes) throws IOException {
        ensureOpen();
        originalInput.skipBytes(numBytes);
    }

    @Override
    public long length() {
        ensureOpen();
        return fileLength;
    }

    @Override
    public void prefetch(final long offset, final long length) throws IOException {
        ensureOpen();
        // This input's own reads are the original input's in both states, so its hints are too.
        originalInput.prefetch(offset, length);
    }

    @Override
    public void updateIOContext(final IOContext context) throws IOException {
        ensureOpen();
        originalInput.updateIOContext(context);
    }

    @Override
    public Optional<Boolean> isLoaded() {
        ensureOpen();
        return originalInput.isLoaded();
    }

    // Closes the original input and, if one was ever opened, this file's O_DIRECT channel, each exactly once. A slice
    // or a clone this input handed out closes nothing, which is Lucene's own rule for both.
    @Override
    public void close() throws IOException {
        final DirectIOFile toClose;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            toClose = directIOFile;
            directIOFile = null;
        }
        IOUtils.close(originalInput, toClose);
    }

    private void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this input is closed: " + this);
        }
    }
}
