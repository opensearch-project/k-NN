/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.misc.store.DirectIODirectory;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.ShardPathUtil;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.store.DirectIOTestUtils.CryptoShapedFsDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.GhostNameFsDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.PassThroughDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannels;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestCompoundDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.store.DirectIOTestUtils.FAISS_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.META_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.QUANTIZED_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.SQ_VEC_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The directory's own work: each eligibility check on its own and a file that passes them all, the file size floor,
 * the buffer sizes, and {@code wrap} on both faiss formats, which hands the delegate the switch of the index whose
 * UUID {@link ShardPathUtil} reads from the shard path.
 */
public class KNNDirectIODirectoryTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 20_000;
    private static final String INDEX_NAME = "target-index";

    private Path shardPath;
    private final AtomicBoolean on = new AtomicBoolean(true);

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
        on.set(true);
    }

    public void testUsesDirectIOForAFaissVectorDataFileThatPassesEveryCheck() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertTrue("the delegate hands out a switch", file instanceof SwitchableIndexInput);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertTrue("a slice taken while the index is on reads with O_DIRECT", slice instanceof DirectIOVectorIndexInput);
                assertEquals("one channel per initialized file", 1, channels.channels().size());
                assertTrue("the content check read the file through that channel", channels.reads() > 0);
            }
        }
    }

    public void testACompoundSegmentGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (Directory compound = new TestCompoundDirectory(in)) {
                assertOriginalInput(compound, DirectIOTestUtils.directory(compound, on::get), VEC_NAME);
            }
        }
    }

    public void testADirectoryWithNoPathGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = new ByteBuffersDirectory()) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), VEC_NAME);
        }
    }

    public void testAMissingFileGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory backing = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(backing, SQ_VEC_NAME, FILE_PAYLOAD_BYTES);
        }
        try (Directory in = new GhostNameFsDirectory(shardPath, VEC_NAME, SQ_VEC_NAME)) {
            assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), VEC_NAME);
        }
    }

    public void testAFileUnderTheFloorGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, 16);
            // The constructor without test seams, so the production floor applies.
            final KNNDirectIODirectory directory = new KNNDirectIODirectory(in, on::get);
            // Being the original directory's own input is the whole statement: it answers every question that input
            // answers, so nothing makes Lucene open the file a second time for a merge or a sequential search.
            // Asserting isEfficientFor and isLoaded against a second input of the same class would assert nothing.
            assertOriginalInput(in, directory, VEC_NAME);
        }
    }

    public void testAnAbsentDirectOptionGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            // A JDK without com.sun.nio.file.ExtendedOpenOption.DIRECT: nothing is wrapped, so the channel factory
            // must never be called.
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, path -> {
                throw new AssertionError("a runtime without the DIRECT option must not open a channel");
            }, false);
            assertOriginalInput(in, directory, VEC_NAME);
        }
    }

    public void testAFileWhoseDirectOpenFailsReadsThroughTheOriginalInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(
                in,
                on::get,
                path -> { throw new IOException("no O_DIRECT here"); }
            );
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                // The file system is a per-file fact, so a file that passed the checks at open is still wrapped: the
                // open failure is learned at the first slice, and the slice is then the original input's own.
                assertTrue(file instanceof SwitchableIndexInput);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertFalse(slice instanceof DirectIOVectorIndexInput);
                final byte[] read = new byte[expected.length];
                slice.seek(0);
                slice.readBytes(read, 0, read.length);
                assertArrayEquals(expected, read);
            }
        }
    }

    public void testDeclinesACryptoShapedNioFsSubclassWithContent() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory plain = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(plain, VEC_NAME, FILE_PAYLOAD_BYTES);
        }
        try (Directory in = new CryptoShapedFsDirectory(shardPath)) {
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertFalse("a layer that transforms bytes stays on the original input", slice instanceof DirectIOVectorIndexInput);
                assertFalse("and so does every later clone", file.clone() instanceof DirectIOVectorIndexInput);
            }
        }
    }

    public void testUsesDirectIOThroughAPassThroughFilterDirectory() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (Directory in = new PassThroughDirectory(bottom)) {
                final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
                try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    final IndexInput slice = file.slice("vector-data", 0, file.length());
                    assertTrue("a pass-through layer is crossed and the file uses Direct I/O", slice instanceof DirectIOVectorIndexInput);
                }
            }
        }
    }

    public void testOffersNothingForAFileThatIsNotVectorData() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            for (final String name : new String[] { META_NAME, QUANTIZED_NAME, FAISS_NAME }) {
                DirectIOTestUtils.writeCodecFile(in, name, FILE_PAYLOAD_BYTES);
                assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), name);
            }
        }
    }

    public void testWrapOnBothFaissFormats() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            for (final String name : new String[] { VEC_NAME, SQ_VEC_NAME }) {
                DirectIOTestUtils.writeCodecFile(in, name, FILE_PAYLOAD_BYTES);
            }
            for (final String suffix : new String[] {
                new NativeEngines990KnnVectorsFormat().getName() + "_0",
                new Faiss1040ScalarQuantizedKnnVectorsFormat().getName() + "_0" }) {
                final SegmentReadState state = new SegmentReadState(in, segmentInfo(in), FieldInfos.EMPTY, IOContext.DEFAULT, suffix);
                final SegmentReadState wrapped = KNNDirectIODirectory.wrap(state);
                assertTrue("the state of " + suffix + " carries the delegate", wrapped.directory instanceof KNNDirectIODirectory);
                assertSame(state.segmentInfo, wrapped.segmentInfo);
            }
            // And the file each format writes is wrapped by that delegate.
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            for (final String name : new String[] { VEC_NAME, SQ_VEC_NAME }) {
                try (IndexInput file = directory.openInput(name, IOContext.DEFAULT)) {
                    assertTrue(name, file instanceof SwitchableIndexInput);
                }
            }
        }
    }

    public void testAnIndexThatIsOffOpensNoChannel() throws IOException {
        on.set(false);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, path -> {
                throw new AssertionError("an index that is off must not open an O_DIRECT channel");
            });
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                final IndexInput clone = file.clone();
                final byte[] read = new byte[expected.length];
                clone.seek(0);
                clone.readBytes(read, 0, read.length);
                assertArrayEquals(expected, read);
                final byte[] fromSlice = new byte[expected.length];
                slice.seek(0);
                slice.readBytes(fromSlice, 0, fromSlice.length);
                assertArrayEquals(expected, fromSlice);
                assertFalse(
                    "neither reads with O_DIRECT",
                    slice instanceof DirectIOVectorIndexInput || clone instanceof DirectIOVectorIndexInput
                );
            }
        }
    }

    public void testTheDelegateIsInstalledOnAnIndexThatIsOff() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        on.set(false);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            assertEquals("the UUID comes from the shard path whatever the setting says", INDEX_UUID, ShardPathUtil.tryGetIndexUuid(in));
            assertSame(in, FilterDirectory.unwrap(DirectIOTestUtils.directory(in, on::get)));
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            // The delegate hands out a switch even while the index is off, which is what lets a later flip reach a
            // segment that is already open, with no close and open of the index.
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertTrue(file instanceof SwitchableIndexInput);
                assertFalse("and it hands out no Direct I/O input", file.clone() instanceof DirectIOVectorIndexInput);
                on.set(true);
                assertTrue("until the index is turned on", file.clone() instanceof DirectIOVectorIndexInput);
            }
        }
    }

    public void testWrapTakesTheUuidFromTheSegmentInfoDirectory() throws IOException {
        try (Directory shardDirectory = DirectIOTestUtils.fsDirectory(shardPath)) {
            try (Directory compound = new TestCompoundDirectory(shardDirectory)) {
                final SegmentReadState state = new SegmentReadState(
                    compound,
                    segmentInfo(shardDirectory),
                    FieldInfos.EMPTY,
                    IOContext.DEFAULT,
                    "NativeEngines990KnnVectorsFormat_0"
                );
                final SegmentReadState wrapped = KNNDirectIODirectory.wrap(state);
                assertNotSame("a compound segment of a known index is still wrapped", state, wrapped);
                assertTrue(wrapped.directory instanceof KNNDirectIODirectory);
                // The delegate's switch reads the setting of the index the shard path names.
                DirectIOTestUtils.publishIndex(clusterService, INDEX_NAME, INDEX_UUID, true, null);
                assertTrue(((KNNDirectIODirectory) wrapped.directory).useDirectIO());
                DirectIOTestUtils.publishIndex(clusterService, INDEX_NAME, INDEX_UUID, false, null);
                assertFalse(((KNNDirectIODirectory) wrapped.directory).useDirectIO());
                assertSame("the wrapped directory is the one the codec was handed", compound, FilterDirectory.unwrap(wrapped.directory));
                assertSame(state.segmentInfo, wrapped.segmentInfo);
                assertSame(state.fieldInfos, wrapped.fieldInfos);
                assertEquals(state.segmentSuffix, wrapped.segmentSuffix);
            }
        }
    }

    public void testWrapReturnsTheCallersStateForAPathWithNoIndexUuid() throws IOException {
        try (Directory in = new ByteBuffersDirectory()) {
            assertNull("a directory with no path has no index UUID", ShardPathUtil.tryGetIndexUuid(in));
            final SegmentReadState state = new SegmentReadState(in, segmentInfo(in), FieldInfos.EMPTY, IOContext.DEFAULT, "x");
            assertSame(state, KNNDirectIODirectory.wrap(state));
        }
    }

    public void testMinFileSizeMatchesLucene() {
        assertEquals(DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT, KNNDirectIODirectory.MIN_FILE_SIZE_BYTES);
        assertEquals(10L * 1024 * 1024, KNNDirectIODirectory.MIN_FILE_SIZE_BYTES);
    }

    public void testBufferSizeForRandomAndSequentialReaders() {
        for (final int blockSize : new int[] { 512, 4096 }) {
            assertEquals("a random reader takes two blocks", 2 * blockSize, KNNDirectIODirectory.bufferSizeFor(false, blockSize));
            assertEquals(
                "a sequential reader takes Lucene's merge buffer",
                KNNDirectIODirectory.SEQUENTIAL_BUFFER_BYTES,
                KNNDirectIODirectory.bufferSizeFor(true, blockSize)
            );
            assertEquals(0, KNNDirectIODirectory.bufferSizeFor(true, blockSize) % blockSize);
            assertEquals(0, KNNDirectIODirectory.bufferSizeFor(false, blockSize) % blockSize);
        }
        // A block larger than the merge buffer still gets whole blocks.
        final int hugeBlock = 1 << 19;
        assertEquals(hugeBlock, KNNDirectIODirectory.bufferSizeFor(true, hugeBlock));
        assertEquals(2 * hugeBlock, KNNDirectIODirectory.bufferSizeFor(false, hugeBlock));
        assertTrue(KNNDirectIODirectory.isSequential(IOContext.DEFAULT.withHints(DataAccessHint.SEQUENTIAL)));
        assertFalse(KNNDirectIODirectory.isSequential(IOContext.DEFAULT));
    }

    // A file that fails a check openInput can make without a device read is the original directory's own input, of the
    // same class and carrying the same bytes, so an index whose files can never use Direct I/O costs nothing.
    private void assertOriginalInput(final Directory original, final KNNDirectIODirectory directory, final String name) throws IOException {
        try (
            IndexInput fromOriginal = original.openInput(name, IOContext.DEFAULT);
            IndexInput fromKnn = directory.openInput(name, IOContext.DEFAULT)
        ) {
            assertEquals(
                "a file that cannot use Direct I/O must be the original directory's own input",
                fromOriginal.getClass(),
                fromKnn.getClass()
            );
            assertArrayEquals(allBytes(fromOriginal), allBytes(fromKnn));
        }
    }

    private static byte[] allBytes(final IndexInput in) throws IOException {
        final IndexInput clone = in.clone();
        clone.seek(0);
        final byte[] bytes = new byte[(int) clone.length()];
        clone.readBytes(bytes, 0, bytes.length);
        return bytes;
    }

    private static SegmentInfo segmentInfo(final Directory dir) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            1,
            false,
            false,
            Codec.getDefault(),
            Collections.emptyMap(),
            new byte[16],
            Map.of(),
            null
        );
    }

}
