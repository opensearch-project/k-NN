/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.tests.mockfile.FilterFileChannel;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.IOFunction;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

/**
 * The one shared helper of the Direct I/O tests, named as k-NN names {@code PrefetchableScorerTestUtils}.
 */
final class DirectIOTestUtils {

    static final String VEC_NAME = "_0_NativeEngines990KnnVectorsFormat_0.vec";
    static final String HALF_FLOAT_VEC_NAME = "_0_NativeEngines990HalfFloatKnnVectorsFormat_0.vec";
    static final String SQ_VEC_NAME = "_0_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec";
    static final String META_NAME = "_0_NativeEngines990KnnVectorsFormat_0.vemf";
    static final String QUANTIZED_NAME = "_0_Faiss1040ScalarQuantizedKnnVectorsFormat_0.veq";
    static final String FAISS_NAME = "_0_NativeEngines990KnnVectorsFormat_0.faiss";

    static final String INDEX_UUID = "V4qM8m9ZTAyRm0Nn0lOAbw";
    static final String CODEC_NAME = "DirectIOTestFile";
    static final int CODEC_VERSION = 1;

    static final long SMALL_FLOOR = 1024L;

    private static volatile Boolean directIOSupported;

    private DirectIOTestUtils() {}

    // The default data path layout, <root>/indices/<uuid>/0/index.
    static Path defaultLayoutShardPath(final Path root, final String uuid) throws IOException {
        return Files.createDirectories(root.resolve("nodes").resolve("0").resolve("indices").resolve(uuid).resolve("0").resolve("index"));
    }

    // A file with a Lucene codec header, a deterministic payload and a codec footer, which is the shape every check
    // and every integrity read needs.
    static byte[] writeCodecFile(final Directory dir, final String name, final int payloadBytes) throws IOException {
        final byte[] id = new byte[16];
        new Random(payloadBytes).nextBytes(id);
        try (IndexOutput out = dir.createOutput(name, IOContext.DEFAULT)) {
            CodecUtil.writeIndexHeader(out, CODEC_NAME, CODEC_VERSION, id, "0");
            final byte[] payload = payload(payloadBytes);
            out.writeBytes(payload, payload.length);
            CodecUtil.writeFooter(out);
        }
        return readAll(dir, name);
    }

    static byte[] payload(final int bytes) {
        final byte[] payload = new byte[bytes];
        new Random(42).nextBytes(payload);
        return payload;
    }

    static byte[] readAll(final Directory dir, final String name) throws IOException {
        try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
            final byte[] bytes = new byte[(int) in.length()];
            in.readBytes(bytes, 0, bytes.length);
            return bytes;
        }
    }

    // The reads a test compares between two inputs, each rendered as text so that one loop covers every type.
    static final List<NamedRead> EVERY_READ = List.of(
        new NamedRead("readByte", input -> String.valueOf(input.readByte())),
        new NamedRead("readBytes", input -> {
            final byte[] values = new byte[64];
            input.readBytes(values, 0, values.length);
            return Arrays.toString(values);
        }),
        new NamedRead("readBytes without the buffer", input -> {
            final byte[] values = new byte[64];
            input.readBytes(values, 0, values.length, false);
            return Arrays.toString(values);
        }),
        new NamedRead("readShort", input -> String.valueOf(input.readShort())),
        new NamedRead("readInt", input -> String.valueOf(input.readInt())),
        new NamedRead("readVInt", input -> String.valueOf(input.readVInt())),
        new NamedRead("readLong", input -> String.valueOf(input.readLong())),
        new NamedRead("readVLong", input -> String.valueOf(input.readVLong())),
        new NamedRead("readInts", input -> {
            final int[] values = new int[16];
            input.readInts(values, 0, values.length);
            return Arrays.toString(values);
        }),
        new NamedRead("readLongs", input -> {
            final long[] values = new long[16];
            input.readLongs(values, 0, values.length);
            return Arrays.toString(values);
        }),
        new NamedRead("readFloats", input -> {
            final float[] values = new float[16];
            input.readFloats(values, 0, values.length);
            return Arrays.toString(values);
        })
    );

    record NamedRead(String name, IOFunction<IndexInput, String> read) {
        String from(final IndexInput input) throws IOException {
            return read.apply(input);
        }
    }

    // Both inputs read from the codec name's length, the field after the magic, which is a valid vInt and vLong.
    static void assertReadsAlike(final NamedRead read, final IndexInput expected, final IndexInput actual) throws IOException {
        expected.seek(Integer.BYTES);
        actual.seek(Integer.BYTES);
        assertEquals(read.name(), read.from(expected), read.from(actual));
        assertEquals(read.name() + " leaves the cursor where the original input does", expected.getFilePointer(), actual.getFilePointer());
    }

    static Directory fsDirectory(final Path path) throws IOException {
        return FSDirectory.open(path);
    }

    static KNNDirectIODirectory directory(final Directory in, final BooleanSupplier directIOEnabled) {
        return directory(in, directIOEnabled, DirectIOFile.DEFAULT_CHANNEL_FACTORY);
    }

    static KNNDirectIODirectory directory(
        final Directory in,
        final BooleanSupplier directIOEnabled,
        final DirectIOFile.ChannelFactory factory
    ) {
        return directory(in, directIOEnabled, factory, DirectIOFile.DIRECT_OPEN_OPTION != null);
    }

    static KNNDirectIODirectory directory(
        final Directory in,
        final BooleanSupplier directIOEnabled,
        final DirectIOFile.ChannelFactory factory,
        final boolean directIOAvailable
    ) {
        return new KNNDirectIODirectory(in, SMALL_FLOOR, directIOEnabled, factory, directIOAvailable);
    }

    static void publishIndex(
        final ClusterService clusterService,
        final String indexName,
        final String uuid,
        final boolean enabled,
        final AtomicInteger stateReads
    ) {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT.id)
            .put(IndexMetadata.SETTING_INDEX_UUID, uuid)
            .put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled)
            .build();
        final IndexMetadata metadata = IndexMetadata.builder(indexName).settings(settings).numberOfShards(1).numberOfReplicas(0).build();
        final ClusterState clusterState = ClusterState.builder(new ClusterName("test"))
            .metadata(Metadata.builder().put(metadata, false).build())
            .build();
        KNNSettings.state().setClusterService(clusterService);
        Mockito.doAnswer(invocation -> {
            if (stateReads != null) {
                stateReads.incrementAndGet();
            }
            return clusterState;
        }).when(clusterService).state();
    }

    // Decided once per test JVM in the two steps of Lucene's TestDirectIODirectory.checkSupported: the production open
    // option is not null, then one probe through the production channel factory. Every test directory comes from
    // createTempDir under one temp root, so one probe answers for all of them.
    static boolean isDirectIOSupported() {
        Boolean answer = directIOSupported;
        if (answer != null) {
            return answer;
        }
        synchronized (DirectIOTestUtils.class) {
            if (directIOSupported != null) {
                return directIOSupported;
            }
            directIOSupported = probeDirectIO();
            return directIOSupported;
        }
    }

    // On the development desktop no Direct I/O test may skip.
    static void assumeDirectIOSupported() {
        assumeTrue("Requires O_DIRECT on this JDK and file system", isDirectIOSupported());
    }

    private static boolean probeDirectIO() {
        if (DirectIOFile.DIRECT_OPEN_OPTION == null) {
            return false;
        }
        try {
            final Path probe = Files.createTempFile(LuceneTestCase.createTempDir("direct-io-probe"), "dio-probe", ".bin");
            try {
                Files.write(probe, payload(8192));
                DirectIOFile.DEFAULT_CHANNEL_FACTORY.open(probe).close();
                return true;
            } finally {
                Files.deleteIfExists(probe);
            }
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    // Hands out TestChannels over real channels, keeps every one it opened, and can be told to make the reads that
    // follow misbehave. The plugin keeps no counters, so this is how a test sees the device reads it makes.
    static final class TestChannels implements DirectIOFile.ChannelFactory {

        private final List<TestChannel> channels = new ArrayList<>();
        private final boolean withPlainChannel;
        private volatile Behaviour behaviour = Behaviour.NORMAL;

        TestChannels() {
            this(false);
        }

        // A plain channel on the same file is the only way a SHORT read can return real bytes: a 100 byte read on an
        // O_DIRECT channel is refused with EINVAL because the length is not a multiple of the block size.
        TestChannels(final boolean withPlainChannel) {
            this.withPlainChannel = withPlainChannel;
        }

        @Override
        public FileChannel open(final Path path) throws IOException {
            final OpenOption direct = DirectIOFile.DIRECT_OPEN_OPTION;
            final FileChannel real = direct == null
                ? FileChannel.open(path, StandardOpenOption.READ)
                : FileChannel.open(path, StandardOpenOption.READ, direct);
            final FileChannel plain = withPlainChannel ? FileChannel.open(path, StandardOpenOption.READ) : null;
            final TestChannel channel = new TestChannel(real, plain, this);
            synchronized (channels) {
                channels.add(channel);
            }
            return channel;
        }

        void behave(final Behaviour behaviour) {
            this.behaviour = behaviour;
        }

        List<TestChannel> channels() {
            synchronized (channels) {
                return List.copyOf(channels);
            }
        }

        TestChannel last() {
            synchronized (channels) {
                return channels.get(channels.size() - 1);
            }
        }

        int reads() {
            int total = 0;
            for (final TestChannel channel : channels()) {
                total += channel.reads().size();
            }
            return total;
        }

        long readBytes() {
            long total = 0;
            for (final TestChannel channel : channels()) {
                for (final Read read : channel.reads()) {
                    total += read.length();
                }
            }
            return total;
        }
    }

    enum Behaviour {
        NORMAL,
        // Fewer bytes than one block, which leaves the requested offset outside what the read covered. A short read that
        // still covers the request is not a failure and the input reads on.
        SHORT,
        THROW,
        // One block of garbage into the buffer, then a throw, which is a read that half happened.
        GARBAGE_THEN_THROW
    }

    record Read(long position, int length) {
    }

    // Lucene's FilterFileChannel forwards every other FileChannel method, so only the positional read and the close
    // are written here.
    static final class TestChannel extends FilterFileChannel {

        static final int SHORT_READ_BYTES = 100;

        private final FileChannel plain;
        private final TestChannels owner;
        private final List<Read> reads = new ArrayList<>();
        private final AtomicInteger closes = new AtomicInteger();

        TestChannel(final FileChannel delegate, final FileChannel plain, final TestChannels owner) {
            super(delegate);
            this.plain = plain;
            this.owner = owner;
        }

        List<Read> reads() {
            synchronized (reads) {
                return List.copyOf(reads);
            }
        }

        int closes() {
            return closes.get();
        }

        @Override
        public int read(final ByteBuffer dst, final long position) throws IOException {
            synchronized (reads) {
                reads.add(new Read(position, dst.remaining()));
            }
            switch (owner.behaviour) {
                case THROW:
                    throw new IOException("injected read failure at " + position);
                case GARBAGE_THEN_THROW: {
                    final int garbage = Math.min(dst.remaining(), 4096);
                    for (int i = 0; i < garbage; i++) {
                        dst.put((byte) 0xFF);
                    }
                    throw new IOException("injected half-done read at " + position);
                }
                case SHORT: {
                    final int limit = Math.min(dst.remaining(), SHORT_READ_BYTES);
                    final ByteBuffer slice = dst.slice();
                    slice.limit(limit);
                    final int read = (plain == null ? delegate : plain).read(slice, position);
                    if (read > 0) {
                        dst.position(dst.position() + read);
                    }
                    return read;
                }
                default:
                    return delegate.read(dst, position);
            }
        }

        @Override
        protected void implCloseChannel() throws IOException {
            closes.incrementAndGet();
            try {
                if (plain != null) {
                    plain.close();
                }
            } finally {
                super.implCloseChannel();
            }
        }
    }

    static final byte XOR_MASK = (byte) 0x5A;

    // Shaped like encryption at rest: the chain still ends in a file system directory and the file still has a real
    // path, but every byte it serves is transformed, so the content check must decline it.
    static final class CryptoShapedFsDirectory extends NIOFSDirectory {

        CryptoShapedFsDirectory(final Path path) throws IOException {
            super(path);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return new XoringIndexInput(super.openInput(name, context));
        }
    }

    // Serves one name out of another file, so that a name the delegate opens resolves to a path with no file on it.
    static final class GhostNameFsDirectory extends NIOFSDirectory {

        private final String ghostName;
        private final String backingName;

        GhostNameFsDirectory(final Path path, final String ghostName, final String backingName) throws IOException {
            super(path);
            this.ghostName = ghostName;
            this.backingName = backingName;
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return super.openInput(ghostName.equals(name) ? backingName : name, context);
        }
    }

    // The shape Lucene hands a codec for a compound segment: neither a FilterDirectory nor an FSDirectory, so the
    // files behind it have no path of their own.
    static final class TestCompoundDirectory extends org.apache.lucene.codecs.CompoundDirectory {

        private final Directory in;

        TestCompoundDirectory(final Directory in) {
            this.in = in;
        }

        @Override
        public void checkIntegrity() throws IOException {
            // nothing to check in a test stand-in
        }

        @Override
        public String[] listAll() throws IOException {
            return in.listAll();
        }

        @Override
        public long fileLength(final String name) throws IOException {
            return in.fileLength(name);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return in.openInput(name, context);
        }

        @Override
        public Set<String> getPendingDeletions() throws IOException {
            return Set.of();
        }

        @Override
        public void close() throws IOException {
            // the delegate belongs to the test
        }
    }

    // A pass-through layer, which the content check must cross and accept: it changes no byte.
    static final class PassThroughDirectory extends FilterDirectory {

        PassThroughDirectory(final Directory in) {
            super(in);
        }
    }

    static final class CloseCountingDirectory extends FilterDirectory {

        private final AtomicInteger closes = new AtomicInteger();
        private final AtomicInteger opens = new AtomicInteger();

        CloseCountingDirectory(final Directory in) {
            super(in);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            opens.incrementAndGet();
            return new CloseCountingIndexInput(super.openInput(name, context), closes);
        }

        int closes() {
            return closes.get();
        }

        int opens() {
            return opens.get();
        }
    }

    // Its clones and slices transform too, which is what makes it a stand-in for a layer that encrypts at rest.
    static final class XoringIndexInput extends IndexInput {

        private final IndexInput in;

        XoringIndexInput(final IndexInput in) {
            super("xoring(" + in + ")");
            this.in = in;
        }

        @Override
        public byte readByte() throws IOException {
            return (byte) (in.readByte() ^ XOR_MASK);
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
            for (int i = offset; i < offset + len; i++) {
                b[i] = (byte) (b[i] ^ XOR_MASK);
            }
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new XoringIndexInput(in.slice(sliceDescription, offset, length));
        }

        @Override
        public IndexInput clone() {
            return new XoringIndexInput(in.clone());
        }
    }

    // Slices and clones count on the same counter, so that a test can prove a clone closes nothing.
    static final class CloseCountingIndexInput extends IndexInput {

        private final IndexInput in;
        private final AtomicInteger closes;

        CloseCountingIndexInput(final IndexInput in, final AtomicInteger closes) {
            super("closeCounting(" + in + ")");
            this.in = in;
            this.closes = closes;
        }

        @Override
        public byte readByte() throws IOException {
            return in.readByte();
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
        }

        @Override
        public void readFloats(final float[] dst, final int offset, final int len) throws IOException {
            in.readFloats(dst, offset, len);
        }

        @Override
        public void readInts(final int[] dst, final int offset, final int len) throws IOException {
            in.readInts(dst, offset, len);
        }

        @Override
        public void readLongs(final long[] dst, final int offset, final int len) throws IOException {
            in.readLongs(dst, offset, len);
        }

        @Override
        public void close() throws IOException {
            closes.incrementAndGet();
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new CloseCountingIndexInput(in.slice(sliceDescription, offset, length), closes);
        }

        @Override
        public IndexInput clone() {
            return new CloseCountingIndexInput(in.clone(), closes);
        }

        @Override
        public RandomAccessInput randomAccessSlice(final long offset, final long length) throws IOException {
            return in.randomAccessSlice(offset, length);
        }

        @Override
        public void prefetch(final long offset, final long length) throws IOException {
            in.prefetch(offset, length);
        }

        @Override
        public Optional<Boolean> isLoaded() {
            return in.isLoaded();
        }

        @Override
        public boolean isEfficientFor(final IOContext context) {
            return in.isEfficientFor(context);
        }
    }

    static final class PrefetchRecorder {

        private final AtomicInteger prefetches = new AtomicInteger();
        private volatile long lastOffset = -1;
        private volatile long lastLength = -1;

        int prefetches() {
            return prefetches.get();
        }

        long lastOffset() {
            return lastOffset;
        }

        long lastLength() {
            return lastLength;
        }
    }

    static final class PrefetchRecordingIndexInput extends IndexInput {

        private final IndexInput in;
        private final PrefetchRecorder recorder;

        PrefetchRecordingIndexInput(final IndexInput in, final PrefetchRecorder recorder) {
            super("prefetchRecording(" + in + ")");
            this.in = in;
            this.recorder = recorder;
        }

        @Override
        public void prefetch(final long offset, final long length) throws IOException {
            recorder.prefetches.incrementAndGet();
            recorder.lastOffset = offset;
            recorder.lastLength = length;
        }

        @Override
        public byte readByte() throws IOException {
            return in.readByte();
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new PrefetchRecordingIndexInput(in.slice(sliceDescription, offset, length), recorder);
        }

        @Override
        public IndexInput clone() {
            return new PrefetchRecordingIndexInput(in.clone(), recorder);
        }
    }

    static final class PrefetchRecordingDirectory extends FilterDirectory {

        private final PrefetchRecorder recorder = new PrefetchRecorder();

        PrefetchRecordingDirectory(final Directory in) {
            super(in);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return new PrefetchRecordingIndexInput(super.openInput(name, context), recorder);
        }

        PrefetchRecorder recorder() {
            return recorder;
        }
    }
}
