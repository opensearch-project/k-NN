/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An {@link IndexInput} over flat vector data that reads with blocking positional reads on the calling thread from a
 * file channel opened with {@code O_DIRECT}, so the bytes do not go through the page cache. Every read is rounded out
 * to whole blocks of the file store's block size into a reusable direct buffer, shared by the random readers on a thread.
 *
 * <p>Beside the channel each instance keeps the matching input of the original directory, with the same coordinates.
 * A read that fails on the channel is read again through that input, from the start of the whole request, and the
 * file's channel is marked failed so that every later read of that file goes there too. So a read failure never fails
 * a read. An {@link EOFException} from a bounds check is thrown before any device read and is not such a failure.
 *
 * <p>Neither the {@link DirectIOFile} nor the original input is closed by this class: both belong to the
 * {@link SwitchableIndexInput} that opened them, which closes each exactly once.
 */
@Log4j2
public final class DirectIOVectorIndexInput extends IndexInput {

    private static final AtomicLong IDS = new AtomicLong();

    private final DirectIOFile file;
    private final IndexInput originalInput;
    private final long offset;
    private final long length;
    private final boolean sequential;
    private final int bufferSize;
    private final long id = IDS.incrementAndGet();

    private long filePointer;
    private boolean closed;
    private DirectIOWindow ownWindow;
    // The buffer the last fill read through, its thread's window or this input's own, and the bytes it holds.
    private ByteBuffer buffer;
    private int bufferValid;

    DirectIOVectorIndexInput(
        final String description,
        @NonNull final DirectIOFile file,
        @NonNull final IndexInput originalInput,
        final long offset,
        final long length,
        final boolean sequential,
        final long filePointer
    ) {
        super(description);
        this.file = file;
        this.originalInput = originalInput;
        this.offset = offset;
        this.length = length;
        this.sequential = sequential;
        this.bufferSize = KNNDirectIODirectory.bufferSizeFor(sequential, file.blockSize());
        this.filePointer = filePointer;
    }

    @Override
    public byte readByte() throws IOException {
        return (byte) readValue(Byte.BYTES);
    }

    @Override
    public short readShort() throws IOException {
        return (short) readValue(Short.BYTES);
    }

    @Override
    public int readInt() throws IOException {
        return (int) readValue(Integer.BYTES);
    }

    @Override
    public long readLong() throws IOException {
        return readValue(Long.BYTES);
    }

    // The bytes of a value of one, two, four or eight bytes, in the low bits of the result, which each caller casts
    // back to its own width.
    private long readValue(final int width) throws IOException {
        ensureOpen();
        if (filePointer + width > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (file.hasFailed() == false) {
            try {
                final long value = valueAt(offset + start, width);
                filePointer = start + width;
                return value;
            } catch (IOException e) {
                fallback(e, offset + start);
            }
        }
        originalInput.seek(start);
        final long value = switch (width) {
            case Byte.BYTES -> originalInput.readByte();
            case Short.BYTES -> originalInput.readShort();
            case Integer.BYTES -> originalInput.readInt();
            default -> originalInput.readLong();
        };
        filePointer = start + width;
        return value;
    }

    @Override
    public void readBytes(final byte[] b, final int bufferOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(bufferOffset, len, b.length);
        if (filePointer + len > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (file.hasFailed() == false) {
            try {
                copyAt(offset + start, b, bufferOffset, len);
                filePointer = start + len;
                return;
            } catch (IOException e) {
                fallback(e, offset + start);
            }
        }
        originalInput.seek(start);
        originalInput.readBytes(b, bufferOffset, len);
        filePointer = start + len;
    }

    @Override
    public void readInts(final int[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final long bytes = (long) len * Integer.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (file.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Integer.BYTES;
                    final int index = fill(absolute);
                    final int available = (bufferValid - index) / Integer.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = (int) valueAt(absolute, Integer.BYTES);
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getInt(index + i * Integer.BYTES);
                    }
                    done += count;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e, offset + start);
            }
        }
        originalInput.seek(start);
        originalInput.readInts(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    @Override
    public void readLongs(final long[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final long bytes = (long) len * Long.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (file.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Long.BYTES;
                    final int index = fill(absolute);
                    final int available = (bufferValid - index) / Long.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = valueAt(absolute, Long.BYTES);
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getLong(index + i * Long.BYTES);
                    }
                    done += count;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e, offset + start);
            }
        }
        originalInput.seek(start);
        originalInput.readLongs(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    @Override
    public void readFloats(final float[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final long bytes = (long) len * Float.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (file.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Float.BYTES;
                    final int index = fill(absolute);
                    final int available = (bufferValid - index) / Float.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = Float.intBitsToFloat((int) valueAt(absolute, Float.BYTES));
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getFloat(index + i * Float.BYTES);
                    }
                    done += count;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e, offset + start);
            }
        }
        originalInput.seek(start);
        originalInput.readFloats(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    // A value that straddles the end of the buffer window is assembled from its bytes, which is the only case the
    // buffer's own little-endian reads cannot serve.
    private long valueAt(final long absolute, final int width) throws IOException {
        final int index = fill(absolute);
        if (bufferValid - index >= width) {
            return switch (width) {
                case Byte.BYTES -> buffer.get(index);
                case Short.BYTES -> buffer.getShort(index);
                case Integer.BYTES -> buffer.getInt(index);
                default -> buffer.getLong(index);
            };
        }
        final byte[] value = new byte[width];
        copyAt(absolute, value, 0, width);
        long assembled = 0L;
        for (int i = width - 1; i >= 0; i--) {
            assembled = (assembled << 8) | (value[i] & 0xFFL);
        }
        return assembled;
    }

    private void fallback(final IOException cause, final long absolute) {
        if (file.markFailed()) {
            log.error(
                "Direct I/O read of {} at offset {} failed, this file is read through the original input until it is opened again",
                file.path(),
                absolute,
                cause
            );
        }
    }

    private void copyAt(final long absolute, final byte[] dst, final int dstOffset, final int len) throws IOException {
        int done = 0;
        while (done < len) {
            final int index = fill(absolute + done);
            final int available = Math.min(len - done, bufferValid - index);
            buffer.get(index, dst, dstOffset + done, available);
            done += available;
        }
    }

    // Makes sure the buffer holds the byte at an absolute offset, with one aligned positional read when it does not,
    // and returns that byte's index in the buffer.
    private int fill(final long absolute) throws IOException {
        final DirectIOWindow window = window();
        buffer = window.buffer(bufferSize, file.blockSize());
        final int inWindow = window.indexOf(id, absolute);
        if (inWindow >= 0) {
            bufferValid = window.valid();
            return inWindow;
        }
        final long alignedStart = file.alignDown(absolute);
        buffer.clear().limit(bufferSize);
        window.clear();
        bufferValid = 0;
        final int read = file.read(buffer, alignedStart);
        if (read <= 0) {
            throw new IOException("Direct I/O read of " + file.path() + " at " + alignedStart + " returned " + read);
        }
        final int index = (int) (absolute - alignedStart);
        if (index >= read) {
            throw new IOException(
                "Direct I/O read of " + file.path() + " at " + alignedStart + " returned " + read + ", need " + (index + 1)
            );
        }
        window.filled(id, alignedStart, read);
        bufferValid = read;
        return index;
    }

    // A random reader reads through its thread's window, a sequential one through its own.
    private DirectIOWindow window() {
        if (sequential == false) {
            return DirectIOWindow.ofThread();
        }
        if (ownWindow == null) {
            ownWindow = new DirectIOWindow();
        }
        return ownWindow;
    }

    @Override
    public long getFilePointer() {
        return filePointer;
    }

    @Override
    public void seek(final long pos) throws IOException {
        ensureOpen();
        if (pos < 0 || pos > length) {
            throw new EOFException("seek past EOF: " + this);
        }
        filePointer = pos;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long sliceOffset, final long sliceLength) throws IOException {
        ensureOpen();
        if ((sliceLength | sliceOffset) < 0 || sliceLength > length - sliceOffset) {
            throw new IllegalArgumentException(
                "slice() " + sliceDescription + " out of bounds: offset=" + sliceOffset + ",length=" + sliceLength + ",fileLength=" + length
            );
        }
        return new DirectIOVectorIndexInput(
            getFullSliceDescription(sliceDescription),
            file,
            originalInput.slice(sliceDescription, sliceOffset, sliceLength),
            offset + sliceOffset,
            sliceLength,
            sequential,
            0L
        );
    }

    @Override
    public DirectIOVectorIndexInput clone() {
        ensureOpen();
        return new DirectIOVectorIndexInput(
            toString() + " [slice=clone]",
            file,
            originalInput.clone(),
            offset,
            length,
            sequential,
            filePointer
        );
    }

    /**
     * Returns the original input's random-access slice. Lucene reads a sparse field's ord-to-doc table and the jump
     * table of its docs-with-field bitmap through this, one value at a time, which here would be a device read per
     * lookup. Both are small next to the vectors.
     */
    @Override
    public RandomAccessInput randomAccessSlice(final long sliceOffset, final long sliceLength) throws IOException {
        ensureOpen();
        return originalInput.randomAccessSlice(sliceOffset, sliceLength);
    }

    @Override
    public void prefetch(final long prefetchOffset, final long prefetchLength) throws IOException {
        ensureOpen();
        // A blocking read on the calling thread would be a read, not a hint.
    }

    @Override
    public Optional<Boolean> isLoaded() {
        return Optional.empty();
    }

    @Override
    public void close() throws IOException {
        closed = true;
        buffer = null;
        ownWindow = null;
    }

    private void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this input is closed: " + this);
        }
    }
}
