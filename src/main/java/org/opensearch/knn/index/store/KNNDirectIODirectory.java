/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.knn.common.ShardPathUtil;
import org.opensearch.knn.index.IndexSettingsRegistry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.function.BooleanSupplier;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_EXTENSION;

/**
 * A {@link FilterDirectory} that the two faiss vector formats put in front of a segment's directory, through
 * {@link #wrap}, so that the flat vector data of a faiss field can be read with {@code O_DIRECT} instead of through
 * the page cache. It decides and configures, as Lucene's {@code DirectIODirectory} does, while
 * {@link SwitchableIndexInput} chooses per slice and clone, {@link DirectIOFile} opens and checks the channel and
 * {@link DirectIOVectorIndexInput} reads.
 *
 * <p>One of these is made per segment per reader, and closing it closes nothing: the directory below belongs to the
 * reader that built the {@link SegmentReadState}, and each {@link SwitchableIndexInput} closes its own channel.
 */
@Log4j2
public final class KNNDirectIODirectory extends FilterDirectory {

    // Lucene's DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT
    static final long MIN_FILE_SIZE_BYTES = 10L * 1024 * 1024;
    // Lucene's DirectIODirectory.DEFAULT_MERGE_BUFFER_SIZE
    static final int SEQUENTIAL_BUFFER_BYTES = 256 * 1024;

    private final long minFileSize;
    private final BooleanSupplier directIOEnabled;
    private final DirectIOFile.ChannelFactory channelFactory;
    private final boolean directIOAvailable;

    /**
     * Puts this delegate in front of the directory of the segment a flat vector reader is about to open.
     *
     * @param state The state the faiss format was handed by Lucene.
     * @return A state whose directory is this delegate, or the given state when the shard path carries no index UUID or
     *         no index on this node has it, because the setting of such a segment can never be read.
     */
    public static SegmentReadState wrap(@NonNull final SegmentReadState state) {
        final String indexUuid = ShardPathUtil.tryGetIndexUuid(state.segmentInfo.dir);
        final Optional<BooleanSupplier> directIOEnabled = StringUtils.isEmpty(indexUuid)
            ? Optional.empty()
            : IndexSettingsRegistry.getInstance().directIOEnabled(indexUuid);
        if (directIOEnabled.isEmpty()) {
            return state;
        }
        return new SegmentReadState(
            new KNNDirectIODirectory(state.directory, directIOEnabled.get()),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }

    KNNDirectIODirectory(@NonNull final Directory in, @NonNull final BooleanSupplier directIOEnabled) {
        this(in, MIN_FILE_SIZE_BYTES, directIOEnabled, DirectIOFile.DEFAULT_CHANNEL_FACTORY, DirectIOFile.DIRECT_OPEN_OPTION != null);
    }

    KNNDirectIODirectory(
        @NonNull final Directory in,
        final long minFileSize,
        @NonNull final BooleanSupplier directIOEnabled,
        @NonNull final DirectIOFile.ChannelFactory channelFactory,
        final boolean directIOAvailable
    ) {
        super(in);
        this.minFileSize = minFileSize;
        this.directIOEnabled = directIOEnabled;
        this.channelFactory = channelFactory;
        this.directIOAvailable = directIOAvailable;
    }

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        if (IndexFileNames.matchesExtension(name, VECTOR_DATA_EXTENSION) == false || directIOAvailable == false) {
            return in.openInput(name, context);
        }
        final IndexInput originalInput = in.openInput(name, context);
        boolean handedOver = false;
        try {
            final IndexInput result = wrapOrDecline(name, context, originalInput);
            handedOver = true;
            return result;
        } finally {
            if (handedOver == false) {
                IOUtils.closeWhileHandlingException(originalInput);
            }
        }
    }

    // The directory below, the shard's or a compound segment's, belongs to the reader that built the state.
    @Override
    public void close() {}

    // The checks that need no device read. A decline here is normal, so it logs at debug.
    private IndexInput wrapOrDecline(final String name, final IOContext context, final IndexInput originalInput) {
        // A compound segment declines here: its directory is Lucene's compound reader, which is not a FilterDirectory,
        // so the chain does not end in a file-system directory.
        final Path path = resolvePath(in, name);
        if (path == null) {
            log.debug("Direct I/O declined for {}: the directory chain does not end in a file-system directory: {}", name, in);
            return originalInput;
        }
        if (Files.isRegularFile(path) == false) {
            log.debug("Direct I/O declined for {}: {} is not a regular file", name, path);
            return originalInput;
        }
        final long length = originalInput.length();
        if (length < minFileSize) {
            log.debug("Direct I/O declined for {}: {} bytes is under the floor of {}", name, length, minFileSize);
            return originalInput;
        }
        return new SwitchableIndexInput(this, name, path, originalInput, length, isSequential(context), channelFactory);
    }

    // Two blocks for a random reader, which hold any vector up to one block even when it straddles a boundary, and
    // Lucene's merge buffer for a sequential one.
    static int bufferSizeFor(final boolean sequential, final int blockSize) {
        final int target = sequential ? SEQUENTIAL_BUFFER_BYTES : 2 * blockSize;
        return DirectIOFile.roundUpToBlock(target, blockSize);
    }

    static boolean isSequential(final IOContext context) {
        return context.context() == IOContext.Context.MERGE || context.hints().contains(DataAccessHint.SEQUENTIAL);
    }

    // Read once per slice, per clone and per isEfficientFor call on a wrapped file, never per read.
    boolean useDirectIO() {
        return directIOEnabled.getAsBoolean();
    }

    // The real path behind a directory chain, or null when the chain does not end in an FSDirectory. The chain is not
    // judged by the class names in it: a layer that transforms bytes is caught by comparing the bytes.
    static Path resolvePath(final Directory dir, final String name) {
        final Directory bottom = FilterDirectory.unwrap(dir);
        return bottom instanceof FSDirectory ? ((FSDirectory) bottom).getDirectory().resolve(name) : null;
    }
}
