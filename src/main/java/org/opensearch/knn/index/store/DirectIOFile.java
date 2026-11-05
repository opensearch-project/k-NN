/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.util.io.IOUtils;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One file's {@code O_DIRECT} channel, shared by every {@link DirectIOVectorIndexInput} over that file and closed by
 * the {@link SwitchableIndexInput} that opened it. {@link #open} opens the channel and checks its bytes against the
 * input the original directory opened, and returns null when the file cannot be read this way.
 *
 * <p>The content check compares the length, the head and the codec footer, so a directory layer that transforms the
 * bytes a reader sees, such as encryption at rest, is declined. <b>A transform of interior bytes only is not
 * detected</b>, and such a layer would serve transformed vectors through this delegate.
 */
@Log4j2
final class DirectIOFile implements Closeable {

    static final int CONTENT_HEADER_BYTES = 4096;

    // Looked up reflectively, as Lucene's DirectIODirectory.ExtendedOpenOption_DIRECT is, so that the plugin links on
    // a JDK without the option. Null then, and no file is ever read with O_DIRECT.
    static final OpenOption DIRECT_OPEN_OPTION = lookUpDirectOption();

    private static OpenOption lookUpDirectOption() {
        try {
            final Class<? extends OpenOption> extended = Class.forName("com.sun.nio.file.ExtendedOpenOption").asSubclass(OpenOption.class);
            final OpenOption[] options = extended.getEnumConstants();
            if (options != null) {
                final OpenOption direct = Arrays.stream(options)
                    .filter(candidate -> "DIRECT".equalsIgnoreCase(candidate.toString()))
                    .findFirst()
                    .orElse(null);
                if (direct != null) {
                    return direct;
                }
            }
        } catch (ClassNotFoundException | RuntimeException e) {
            // The plugin runs without Direct I/O rather than failing to link.
        }
        log.warn(
            "Direct I/O is not available on this runtime: com.sun.nio.file.ExtendedOpenOption.DIRECT is absent, so "
                + "index.knn.direct_io.rescore.enabled has no effect and vector data is read through the page cache"
        );
        return null;
    }

    @FunctionalInterface
    interface ChannelFactory {
        FileChannel open(Path path) throws IOException;
    }

    static final ChannelFactory DEFAULT_CHANNEL_FACTORY = DirectIOFile::openDirectChannel;

    private final Path path;
    private final FileChannel channel;
    private final int blockSize;
    private final long fileLength;
    private final AtomicBoolean failed = new AtomicBoolean();

    private DirectIOFile(final Path path, final FileChannel channel, final int blockSize, final long fileLength) {
        this.path = path;
        this.channel = channel;
        this.blockSize = blockSize;
        this.fileLength = fileLength;
    }

    // The two checks that need the device: the O_DIRECT open and the byte comparison against the directory chain.
    // Null when either declines, so that the caller serves the original input.
    static DirectIOFile open(
        @NonNull final Path path,
        final long length,
        @NonNull final IndexInput original,
        @NonNull final ChannelFactory channelFactory
    ) {
        final DirectIOFile opened = openDirect(path, length, channelFactory);
        if (opened == null) {
            return null;
        }
        if (contentMatches(opened, original)) {
            return opened;
        }
        IOUtils.closeWhileHandlingException(opened);
        // The only signal an operator gets that a layer in the chain transforms the bytes a reader sees, so it is a
        // warning rather than a debug line.
        log.warn("Direct I/O declined for {}: the bytes read through the directory chain and through O_DIRECT differ", path);
        return null;
    }

    // Returns null rather than throwing when O_DIRECT is not available for this file, so that the caller serves the
    // original input. A file that passed the checks at open was expected to use Direct I/O, so the decline is a
    // warning.
    private static DirectIOFile openDirect(final Path path, final long length, final ChannelFactory channelFactory) {
        FileChannel channel = null;
        try {
            final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());
            if (blockSize <= 0 || Integer.bitCount(blockSize) != 1) {
                // A buffer can only be aligned to a power of two, so anything else declines here rather than
                // surfacing later as a content mismatch.
                log.warn("Direct I/O declined for {}: the file store reports a block size of {}, not a power of two", path, blockSize);
                return null;
            }
            channel = channelFactory.open(path);
            return new DirectIOFile(path, channel, blockSize, length);
        } catch (IOException | RuntimeException e) {
            IOUtils.closeWhileHandlingException(channel);
            log.warn("Direct I/O declined for {}: the file cannot be opened with O_DIRECT: {}", path, e.toString());
            return null;
        }
    }

    // The last check: the same length, head and codec footer through the O_DIRECT channel and through the directory
    // chain. The original input is left where it was.
    private static boolean contentMatches(final DirectIOFile file, final IndexInput original) {
        try {
            if (original.length() != file.channel.size()) {
                // The file on disk, measured through the O_DIRECT channel, against the length the directory chain
                // reports. Comparing against the length field would compare the original input's length with a copy
                // of itself.
                return false;
            }
            final int headerBytes = (int) Math.min((long) CONTENT_HEADER_BYTES, file.fileLength);
            if (headerBytes <= 0 || regionMatches(file, original, 0L, headerBytes) == false) {
                return false;
            }
            final long footerStart = file.fileLength - CodecUtil.footerLength();
            if (footerStart < headerBytes) {
                // The header comparison already covered the footer.
                return true;
            }
            return regionMatches(file, original, footerStart, CodecUtil.footerLength());
        } catch (IOException | RuntimeException e) {
            log.warn("Direct I/O declined for {}: the content check failed: {}", file.path, e.toString());
            return false;
        }
    }

    private static boolean regionMatches(final DirectIOFile file, final IndexInput original, final long start, final int length)
        throws IOException {
        final byte[] fromChannel = readAt(file, start, length);
        final byte[] fromOriginal = new byte[length];
        final IndexInput clone = original.clone();
        clone.seek(start);
        clone.readBytes(fromOriginal, 0, length);
        return Arrays.equals(fromChannel, fromOriginal);
    }

    // One block-aligned positional read of a region into a fresh buffer, which only the content check uses.
    private static byte[] readAt(final DirectIOFile file, final long start, final int length) throws IOException {
        final long alignedStart = file.alignDown(start);
        final int span = Math.toIntExact(start - alignedStart) + length;
        final ByteBuffer buffer = allocateAligned(roundUpToBlock(span, file.blockSize), file.blockSize);
        final int read = file.read(buffer, alignedStart);
        if (read < span) {
            throw new IOException("Direct I/O read of " + file.path + " at " + alignedStart + " returned " + read + ", need " + span);
        }
        final byte[] bytes = new byte[length];
        buffer.get(Math.toIntExact(start - alignedStart), bytes, 0, length);
        return bytes;
    }

    static int roundUpToBlock(final int bytes, final int blockSize) {
        return ((bytes + blockSize - 1) / blockSize) * blockSize;
    }

    static ByteBuffer allocateAligned(final int bytes, final int blockSize) {
        return ByteBuffer.allocateDirect(bytes + blockSize - 1).alignedSlice(blockSize).order(ByteOrder.LITTLE_ENDIAN).limit(bytes);
    }

    private static FileChannel openDirectChannel(final Path path) throws IOException {
        return FileChannel.open(path, StandardOpenOption.READ, DIRECT_OPEN_OPTION);
    }

    // True on the one call that sets the mark, so that the failure is logged once per file per open.
    boolean markFailed() {
        return failed.compareAndSet(false, true);
    }

    boolean hasFailed() {
        return failed.get();
    }

    int blockSize() {
        return blockSize;
    }

    Path path() {
        return path;
    }

    long alignDown(final long absolute) {
        return absolute - Math.floorMod(absolute, (long) blockSize);
    }

    int read(final ByteBuffer target, final long position) throws IOException {
        return channel.read(target, position);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
