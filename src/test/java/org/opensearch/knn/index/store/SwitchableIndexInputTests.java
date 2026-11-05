/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.RandomAccessInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.store.DirectIOTestUtils.CloseCountingDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.NamedRead;
import org.opensearch.knn.index.store.DirectIOTestUtils.PrefetchRecordingDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannel;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannels;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The switch itself: the setting read when a slice or a clone is made and never per read, a slice keeping the input it
 * was made with, Direct I/O initialized once however often the switch flips, the two hint rules, the switch's share of
 * the close rules, and the reflection test that every method Lucene's mmap input overrides is forwarded.
 */
public class SwitchableIndexInputTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 32 * 1024;
    private static final int CYCLES = 50;
    private static final String INDEX_NAME = "target-index";

    private Path shardPath;
    private final AtomicBoolean on = new AtomicBoolean(false);
    private final AtomicInteger stateReads = new AtomicInteger();

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
        on.set(false);
        stateReads.set(0);
    }

    // Every case but the reflection test needs a real O_DIRECT channel. That one compares two reflection surfaces and
    // must run on every platform, because it is the guard against a later Lucene adding a method to the mmap input.
    private void assumeDirectIO() {
        DirectIOTestUtils.assumeDirectIOSupported();
    }

    public void testASliceTakenAfterTheFlipToOnReadsWithDirectIO() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                on.set(true);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertTrue(slice instanceof DirectIOVectorIndexInput);
                final int afterInitialize = channels.reads();
                final byte[] read = new byte[4096];
                slice.seek(0);
                slice.readBytes(read, 0, read.length);
                for (int i = 0; i < read.length; i++) {
                    assertEquals("byte " + i, expected[i], read[i]);
                }
                assertTrue("the read went to the device", channels.reads() > afterInitialize);
            }
        }
    }

    public void testASliceTakenAfterTheFlipToOffIsTheOriginalInputsOwn() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                on.set(true);
                file.clone();
                final int readsAfterOn = channels.reads();
                on.set(false);
                final IndexInput offSlice = file.slice("vector-data", 0, file.length());
                assertFalse(offSlice instanceof DirectIOVectorIndexInput);
                final byte[] read = new byte[4096];
                offSlice.seek(0);
                offSlice.readBytes(read, 0, read.length);
                for (int i = 0; i < read.length; i++) {
                    assertEquals("byte " + i, expected[i], read[i]);
                }
                assertEquals("an off slice makes no device read", readsAfterOn, channels.reads());
            }
        }
    }

    public void testASliceKeepsTheInputItWasMadeWith() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                on.set(true);
                final IndexInput directIOSlice = file.slice("made-while-on", 0, file.length());
                on.set(false);
                final byte[] target = new byte[2048];
                final int before = channels.reads();
                directIOSlice.seek(0);
                directIOSlice.readBytes(target, 0, target.length);
                assertTrue("a slice made while on keeps reading with O_DIRECT", channels.reads() > before);
                assertFalse("the next slice reads the new value", file.clone() instanceof DirectIOVectorIndexInput);
            }
        }
    }

    public void testTheFirstSliceWhileOnInitializesDirectIoOnce() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                for (int i = 0; i < 5; i++) {
                    on.set(true);
                    file.clone();
                    file.slice("flip-" + i, 0, 64);
                    on.set(false);
                    file.clone();
                }
                on.set(true);
                assertTrue("Direct I/O initialized and the clone reads through it", file.clone() instanceof DirectIOVectorIndexInput);
                assertEquals("a file opens one channel however often the switch flips", 1, channels.channels().size());
            }
        }
    }

    public void testTheSettingIsReadOnEverySliceAndCloneAndNeverPerRead() throws IOException {
        assumeDirectIO();
        // On, so that the clone under test reads with O_DIRECT and the reads below are the ones that could have
        // looked the setting up.
        DirectIOTestUtils.publishIndex(clusterService, INDEX_NAME, INDEX_UUID, true, stateReads);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            // The production switch, which is what this test is about, with the file size floor lowered so the test
            // file can use Direct I/O: a declined file would hand out the original input's clone and the read assertion
            // below would hold for the wrong reason.
            final KNNDirectIODirectory directory = new KNNDirectIODirectory(
                in,
                DirectIOTestUtils.SMALL_FLOOR,
                () -> KNNSettings.isDirectIORescoreEnabled(INDEX_UUID),
                DirectIOFile.DEFAULT_CHANNEL_FACTORY,
                DirectIOFile.DIRECT_OPEN_OPTION != null
            );
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertEquals("nothing is read from the cluster state at open", 0, stateReads.get());
                final IndexInput clone = file.clone();
                assertTrue(
                    "the clone must read with O_DIRECT, or the reads below prove nothing",
                    clone instanceof DirectIOVectorIndexInput
                );
                assertEquals("one cluster state read per clone", 1, stateReads.get());
                final byte[] target = new byte[1024];
                for (int i = 0; i < 20; i++) {
                    clone.seek(0);
                    clone.readBytes(target, 0, target.length);
                }
                assertEquals("a read never looks the setting up", 1, stateReads.get());
                file.clone();
                assertEquals("and the next clone reads it once more", 2, stateReads.get());
            }
        }
    }

    public void testEverySliceAndCloneWhileOffIsTheOriginalInputsOwnObject() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertFalse(file.slice("a", 0, 64) instanceof DirectIOVectorIndexInput);
                assertFalse(file.slice("b", 0, 64, IOContext.DEFAULT) instanceof DirectIOVectorIndexInput);
                assertFalse(file.clone() instanceof DirectIOVectorIndexInput);
                assertNotNull("the ord-to-doc table reads through the original input", file.randomAccessSlice(0, 64));
            }
        }
    }

    public void testPrefetchDoesNothingOnADirectIOSlice() throws IOException {
        assumeDirectIO();
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final PrefetchRecordingDirectory recording = new PrefetchRecordingDirectory(bottom);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(recording, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                on.set(true);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                final int afterInitialize = channels.reads();
                slice.prefetch(0, 4096);
                assertEquals("a Direct I/O input does not carry out a prefetch", 0, recording.recorder().prefetches());
                assertEquals("and does not read either", afterInitialize, channels.reads());
            }
        }
    }

    public void testPrefetchReachesTheOriginalInputFromTheForwarderAndFromAnOffSlice() throws IOException {
        assumeDirectIO();
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final PrefetchRecordingDirectory recording = new PrefetchRecordingDirectory(bottom);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(recording, on::get);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                file.seek(128);
                file.prefetch(512, 4096);
                assertEquals(1, recording.recorder().prefetches());
                assertEquals(512L, recording.recorder().lastOffset());
                assertEquals(4096L, recording.recorder().lastLength());
                assertEquals("a prefetch does not move the cursor", 128L, file.getFilePointer());

                final IndexInput offSlice = file.slice("vector-data", 0, file.length());
                offSlice.prefetch(1024, 2048);
                assertEquals(2, recording.recorder().prefetches());
                assertEquals(1024L, recording.recorder().lastOffset());
            }
        }
    }

    public void testIsEfficientForIsTheOriginalInputsAnswerWhileOff() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            final IOContext sequential = IOContext.DEFAULT.withHints(DataAccessHint.SEQUENTIAL);
            try (
                IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
                IndexInput reference = in.openInput(VEC_NAME, IOContext.DEFAULT)
            ) {
                assertEquals(reference.isEfficientFor(sequential), file.isEfficientFor(sequential));
                assertEquals(reference.isEfficientFor(IOContext.DEFAULT), file.isEfficientFor(IOContext.DEFAULT));
                on.set(true);
                assertFalse("a random Direct I/O input is not efficient for a sequential read", file.isEfficientFor(sequential));
                assertTrue(file.isEfficientFor(IOContext.DEFAULT));
            }
            try (IndexInput sequentialFile = directory.openInput(VEC_NAME, sequential)) {
                assertTrue("a file opened sequentially reads through the larger buffer", sequentialFile.isEfficientFor(sequential));
            }
        }
    }

    public void testRandomAccessSliceIsTheOriginalInputsInBothStates() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertEquals(expected[64], file.randomAccessSlice(0, 4096).readByte(64));
                on.set(true);
                final IndexInput directIO = file.clone();
                assertTrue(directIO instanceof DirectIOVectorIndexInput);
                final int afterInitialize = channels.reads();
                final RandomAccessInput table = directIO.randomAccessSlice(0, 4096);
                assertEquals(expected[64], table.readByte(64));
                assertEquals("the ord-to-doc table is read through the original input in both states", afterInitialize, channels.reads());
            }
        }
    }

    public void testEveryReadOfTheFileItselfIsTheOriginalInputsInBothStates() throws IOException {
        assumeDirectIO();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (
                IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
                IndexInput reference = in.openInput(VEC_NAME, IOContext.DEFAULT)
            ) {
                for (final boolean state : new boolean[] { false, true }) {
                    on.set(state);
                    // While on, this opens the file's channel, so a read below that used it would show in the count.
                    assertEquals(state, file.clone() instanceof DirectIOVectorIndexInput);
                    final int readsBefore = channels.reads();
                    for (final NamedRead read : DirectIOTestUtils.EVERY_READ) {
                        DirectIOTestUtils.assertReadsAlike(read, reference, file);
                    }
                    file.skipBytes(37);
                    reference.skipBytes(37);
                    file.updateIOContext(IOContext.DEFAULT);
                    assertEquals(reference.getFilePointer(), file.getFilePointer());
                    assertEquals(reference.isLoaded(), file.isLoaded());
                    assertEquals("the file's own reads never reach its channel", readsBefore, channels.reads());
                }
            }
        }
    }

    public void testCloseClosesTheChannelAndTheOriginalInputExactlyOnce() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final CloseCountingDirectory counting = new CloseCountingDirectory(bottom);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(counting, on::get, channels);
            final IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            final IndexInput directIO = file.clone();
            assertTrue(directIO instanceof DirectIOVectorIndexInput);
            assertEquals(0, counting.closes());
            assertEquals(0, channels.last().closes());
            file.close();
            assertEquals("the original input is closed once", 1, counting.closes());
            assertEquals("the channel is closed once", 1, channels.last().closes());
        }
    }

    public void testClosingADirectIOCloneClosesNothing() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final CloseCountingDirectory counting = new CloseCountingDirectory(bottom);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(counting, on::get, channels);
            final IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            final IndexInput directIO = file.clone();
            directIO.close();
            assertEquals("a clone closes neither the channel nor the original input", 0, counting.closes());
            assertEquals(0, channels.last().closes());
            file.close();
            assertEquals(1, counting.closes());
            assertEquals(1, channels.last().closes());
        }
    }

    public void testClosingADirectIOSliceClosesNothing() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final CloseCountingDirectory counting = new CloseCountingDirectory(bottom);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(counting, on::get, channels);
            final IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            final IndexInput slice = file.slice("vector-data", 0, file.length());
            slice.close();
            assertEquals(0, counting.closes());
            assertEquals(0, channels.last().closes());
            file.close();
            assertEquals(1, counting.closes());
            assertEquals(1, channels.last().closes());
        }
    }

    public void testCloseIsIdempotent() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final CloseCountingDirectory counting = new CloseCountingDirectory(bottom);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(counting, on::get, channels);
            final IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            file.clone();
            file.close();
            file.close();
            file.close();
            assertEquals(1, counting.closes());
            assertEquals(1, channels.last().closes());
        }
    }

    public void testReadAfterCloseFails() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(bottom, on::get, new TestChannels());
            final IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            final IndexInput directIO = file.clone();
            final byte[] target = new byte[16];
            directIO.close();
            expectThrows(AlreadyClosedException.class, () -> directIO.readByte());
            expectThrows(AlreadyClosedException.class, () -> directIO.readBytes(target, 0, target.length));
            expectThrows(AlreadyClosedException.class, () -> directIO.readFloats(new float[4], 0, 4));
            expectThrows(AlreadyClosedException.class, () -> directIO.seek(0));
            expectThrows(AlreadyClosedException.class, () -> directIO.prefetch(0, 16));
            file.close();
            expectThrows(AlreadyClosedException.class, () -> file.readByte());
        }
    }

    public void testManyOpenAndCloseCyclesLeaveNoChannelOpen() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            for (int i = 0; i < CYCLES; i++) {
                try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    final IndexInput directIO = file.clone();
                    final byte[] target = new byte[64];
                    directIO.seek(0);
                    directIO.readBytes(target, 0, target.length);
                }
            }
            assertEquals("every cycle opened one channel", CYCLES, channels.channels().size());
            for (final TestChannel channel : channels.channels()) {
                assertEquals("each channel is closed exactly once", 1, channel.closes());
                assertFalse(channel.isOpen());
            }
        }
    }

    public void testClosingAFileDoesNotCloseTheDirectory() throws IOException {
        assumeDirectIO();
        on.set(true);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            final long expectedLength = FILE_PAYLOAD_BYTES + CodecUtil.footerLength() + CodecUtil.indexHeaderLength(
                DirectIOTestUtils.CODEC_NAME,
                "0"
            );
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertEquals(expectedLength, file.length());
                file.clone();
            }
            // The delegate directory belongs to the reader that built the segment read state, and the switch holds no
            // resource of its own, so one file closing leaves the directory able to serve the file again.
            try (IndexInput again = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final byte[] head = new byte[64];
                again.clone().readBytes(head, 0, head.length);
                assertEquals(expectedLength, again.length());
            }
            assertEquals("each open initializes its own file", 2, channels.channels().size());
        }
    }

    // Every method Lucene's memory-segment input overrides out of what IndexInput and DataInput declare must be
    // forwarded by this class, or an index that is off would behave differently from main. A method a later Lucene
    // adds fails here.
    public void testEveryMethodTheMmapInputOverridesIsForwarded() throws IOException {
        final Path path = createTempDir("mmap-surface");
        try (Directory mmap = new MMapDirectory(path)) {
            DirectIOTestUtils.writeCodecFile(mmap, VEC_NAME, 4096);
            final Class<?> mmapInput;
            try (IndexInput in = mmap.openInput(VEC_NAME, IOContext.DEFAULT)) {
                mmapInput = in.getClass();
            }
            final Set<String> publicSurface = new LinkedHashSet<>();
            for (final Class<?> surface : new Class<?>[] { IndexInput.class, DataInput.class }) {
                for (final Method method : surface.getDeclaredMethods()) {
                    publicSurface.add(signature(method));
                }
            }
            final Set<String> forwarded = new LinkedHashSet<>();
            for (final Method method : SwitchableIndexInput.class.getDeclaredMethods()) {
                forwarded.add(signature(method));
            }
            final Set<String> missing = new LinkedHashSet<>();
            final Set<String> candidates = new LinkedHashSet<>();
            for (Class<?> current = mmapInput; current != null && current != IndexInput.class; current = current.getSuperclass()) {
                for (final Method method : current.getDeclaredMethods()) {
                    if (method.isSynthetic() || method.isBridge() || Modifier.isStatic(method.getModifiers())) {
                        continue;
                    }
                    if (Modifier.isPublic(method.getModifiers()) == false) {
                        continue;
                    }
                    final String signature = signature(method);
                    if (publicSurface.contains(signature) == false) {
                        continue;
                    }
                    candidates.add(signature);
                    if (forwarded.contains(signature) == false) {
                        missing.add(signature);
                    }
                }
            }
            assertTrue("SwitchableIndexInput does not forward what " + mmapInput.getName() + " overrides: " + missing, missing.isEmpty());
            // The walk itself has to find methods, or an empty missing set would mean nothing.
            assertFalse(
                "the walk over " + mmapInput.getName() + " found no method of the public input surface, so it proves nothing",
                candidates.isEmpty()
            );
            assertTrue("the mmap input overrides more than a handful of them, " + candidates, candidates.size() > 10);
        }
    }

    private static String signature(final Method method) {
        return method.getName() + Arrays.toString(method.getParameterTypes());
    }

}
