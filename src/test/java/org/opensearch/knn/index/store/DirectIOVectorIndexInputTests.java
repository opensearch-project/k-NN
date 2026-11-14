/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.store.DirectIOTestUtils.Behaviour;
import org.opensearch.knn.index.store.DirectIOTestUtils.NamedRead;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannels;

import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The aligned reads at every interesting offset through every read method a flat vector reader uses, the integrity
 * reads, and the fallback to the original input when an {@code O_DIRECT} read fails.
 */
public class DirectIOVectorIndexInputTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 64 * 1024;

    private Path shardPath;
    private final AtomicBoolean on = new AtomicBoolean(true);
    private TestChannels channels;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
        on.set(true);
        // With a plain channel beside each O_DIRECT one, because a short read of 100 bytes on an O_DIRECT channel is
        // refused with EINVAL and would be indistinguishable from the throwing case.
        channels = new TestChannels(true);
        DirectIOTestUtils.assumeDirectIOSupported();
    }

    /** A Direct I/O input over a fresh file, with a counting channel under it, and the reference input beside it. */
    private final class Fixture implements AutoCloseable {
        private final Directory in;
        private final KNNDirectIODirectory directory;
        private final IndexInput file;
        private final IndexInput directIO;
        private final IndexInput reference;
        private final int readsAtInitialize;

        private Fixture() throws IOException {
            in = DirectIOTestUtils.fsDirectory(shardPath);
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            directory = DirectIOTestUtils.directory(in, on::get, channels);
            file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            directIO = file.clone();
            // Without this every read below would compare the original input against itself and pass.
            assertTrue("the clone under test must read with O_DIRECT", directIO instanceof DirectIOVectorIndexInput);
            reference = in.openInput(VEC_NAME, IOContext.DEFAULT);
            readsAtInitialize = channels.reads();
        }

        private int blockSize() throws IOException {
            return Math.toIntExact(Files.getFileStore(shardPath.resolve(VEC_NAME)).getBlockSize());
        }

        @Override
        public void close() throws IOException {
            reference.close();
            file.close();
            in.close();
        }
    }

    public void testReadsBytesAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final long length = fixture.directIO.length();
            for (final long offset : new long[] { 0, block - 1, block, block + 1, 2L * block - 7, length - 32 }) {
                for (final int len : new int[] { 1, 7, 4096, 2 * block + 11 }) {
                    if (offset + len > length) {
                        continue;
                    }
                    final byte[] fromDirectIO = new byte[len];
                    final byte[] fromReference = new byte[len];
                    fixture.directIO.seek(offset);
                    fixture.directIO.readBytes(fromDirectIO, 0, len);
                    fixture.reference.seek(offset);
                    fixture.reference.readBytes(fromReference, 0, len);
                    assertArrayEquals("readBytes at " + offset + " of " + len, fromReference, fromDirectIO);
                    assertEquals(offset + len, fixture.directIO.getFilePointer());
                }
            }
        }
    }

    public void testReadsFloatsAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            for (final long offset : new long[] { 0, block - 4, block, block + 4, 2L * block - 8 }) {
                for (final int floats : new int[] { 1, 128, 1024, block }) {
                    if (offset + 4L * floats > fixture.directIO.length()) {
                        continue;
                    }
                    final float[] fromDirectIO = new float[floats];
                    final float[] fromReference = new float[floats];
                    fixture.directIO.seek(offset);
                    fixture.directIO.readFloats(fromDirectIO, 0, floats);
                    fixture.reference.seek(offset);
                    fixture.reference.readFloats(fromReference, 0, floats);
                    assertArrayEquals("readFloats at " + offset + " of " + floats, fromReference, fromDirectIO, 0.0f);
                }
            }
        }
    }

    public void testReadsTheIntegerTypesAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            for (final long offset : new long[] { 0, block - 1, block, block + 1, 4L * block - 3 }) {
                fixture.directIO.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readByte at " + offset, fixture.reference.readByte(), fixture.directIO.readByte());
                fixture.directIO.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readShort at " + offset, fixture.reference.readShort(), fixture.directIO.readShort());
                fixture.directIO.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readInt at " + offset, fixture.reference.readInt(), fixture.directIO.readInt());
                fixture.directIO.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readLong at " + offset, fixture.reference.readLong(), fixture.directIO.readLong());
                for (final int count : new int[] { 1, 16, 512 }) {
                    if ((long) count * Long.BYTES > fixture.reference.length() - offset) {
                        continue;
                    }
                    final int[] directIOInts = new int[count];
                    final int[] referenceInts = new int[count];
                    fixture.directIO.seek(offset);
                    fixture.reference.seek(offset);
                    fixture.directIO.readInts(directIOInts, 0, count);
                    fixture.reference.readInts(referenceInts, 0, count);
                    assertArrayEquals("readInts of " + count + " at " + offset, referenceInts, directIOInts);
                    assertEquals("the pointer after readInts", fixture.reference.getFilePointer(), fixture.directIO.getFilePointer());
                    final long[] directIOLongs = new long[count];
                    final long[] referenceLongs = new long[count];
                    fixture.directIO.seek(offset);
                    fixture.reference.seek(offset);
                    fixture.directIO.readLongs(directIOLongs, 0, count);
                    fixture.reference.readLongs(referenceLongs, 0, count);
                    assertArrayEquals("readLongs of " + count + " at " + offset, referenceLongs, directIOLongs);
                    assertEquals("the pointer after readLongs", fixture.reference.getFilePointer(), fixture.directIO.getFilePointer());
                }
            }
        }
    }

    public void testReadsLongerThanTheBuffer() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int len = 8 * fixture.blockSize() + 37;
            final byte[] fromDirectIO = new byte[len];
            final byte[] fromReference = new byte[len];
            fixture.directIO.seek(11);
            fixture.directIO.readBytes(fromDirectIO, 0, len);
            fixture.reference.seek(11);
            fixture.reference.readBytes(fromReference, 0, len);
            assertArrayEquals(fromReference, fromDirectIO);
        }
    }

    public void testSliceReadsAreOffsetCorrectly() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final long sliceOffset = block + 17;
            final long sliceLength = 3L * block;
            final IndexInput slice = fixture.file.slice("vector-data", sliceOffset, sliceLength);
            assertTrue(slice instanceof DirectIOVectorIndexInput);
            assertEquals(sliceLength, slice.length());
            final byte[] fromSlice = new byte[256];
            final byte[] fromReference = new byte[256];
            slice.seek(block);
            slice.readBytes(fromSlice, 0, fromSlice.length);
            fixture.reference.seek(sliceOffset + block);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals(fromReference, fromSlice);
        }
    }

    public void testASliceOfADirectIOInputReadsItsOwnRangeAndChecksItsBounds() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final long outerOffset = block + 17;
            final long innerOffset = 13;
            final IndexInput outer = fixture.file.slice("outer", outerOffset, 4L * block);
            assertTrue(outer instanceof DirectIOVectorIndexInput);
            final byte[] fromSlice = new byte[256];
            final byte[] fromReference = new byte[256];
            fixture.reference.seek(outerOffset + innerOffset + block - 100);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            for (final Behaviour behaviour : new Behaviour[] { Behaviour.NORMAL, Behaviour.THROW }) {
                channels.behave(behaviour);
                final IndexInput inner = outer.slice("inner", innerOffset, 2L * block);
                assertTrue(inner instanceof DirectIOVectorIndexInput);
                assertEquals(2L * block, inner.length());
                final int readsBefore = channels.reads();
                inner.seek(block - 100);
                inner.readBytes(fromSlice, 0, fromSlice.length);
                assertArrayEquals(behaviour + " reads the inner slice's own range", fromReference, fromSlice);
                assertEquals("one device read was made or attempted", readsBefore + 1, channels.reads());
            }
            expectThrows(IllegalArgumentException.class, () -> outer.slice("before the start", -1, 16));
            expectThrows(IllegalArgumentException.class, () -> outer.slice("a negative length", 0, -1));
            expectThrows(IllegalArgumentException.class, () -> outer.slice("past the end", 1, outer.length()));
        }
    }

    public void testADirectIOInputDoesNotKnowWhatIsLoaded() throws IOException {
        try (Fixture fixture = new Fixture()) {
            assertEquals("its reads bypass the page cache that isLoaded reports on", Optional.empty(), fixture.directIO.isLoaded());
        }
    }

    public void testCloneHasItsOwnCursor() throws IOException {
        try (Fixture fixture = new Fixture()) {
            fixture.directIO.seek(1024);
            final IndexInput clone = fixture.directIO.clone();
            assertEquals(1024, clone.getFilePointer());
            clone.seek(4096);
            assertEquals("the clone's cursor is its own", 1024, fixture.directIO.getFilePointer());
            assertEquals(4096, clone.getFilePointer());
        }
    }

    public void testReadPastEofThrowsEofAndIsNotAFailure() throws IOException {
        try (Fixture fixture = new Fixture()) {
            fixture.directIO.seek(fixture.directIO.length() - 4);
            final byte[] target = new byte[64];
            final int readsBefore = channels.reads();
            expectThrows(EOFException.class, () -> fixture.directIO.readBytes(target, 0, target.length));
            for (final NamedRead read : DirectIOTestUtils.EVERY_READ) {
                fixture.directIO.seek(fixture.directIO.length());
                expectThrows(EOFException.class, () -> read.from(fixture.directIO));
            }
            assertEquals("the bounds check is thrown before any device read", readsBefore, channels.reads());
            fixture.directIO.seek(0);
            fixture.reference.seek(0);
            assertEquals(fixture.reference.readByte(), fixture.directIO.readByte());
            assertEquals("the next read still goes to the device", readsBefore + 1, channels.reads());
        }
    }

    public void testSeekPastEofThrowsEof() throws IOException {
        try (Fixture fixture = new Fixture()) {
            expectThrows(EOFException.class, () -> fixture.directIO.seek(fixture.directIO.length() + 1));
            expectThrows(EOFException.class, () -> fixture.directIO.seek(-1));
        }
    }

    public void testEveryDeviceReadIsBlockAligned() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final byte[] target = new byte[3 * block + 5];
            fixture.directIO.seek(block - 3);
            fixture.directIO.readBytes(target, 0, target.length);
            assertFalse("the read must have reached the device", channels.last().reads().isEmpty());
            for (final DirectIOTestUtils.Read read : channels.last().reads()) {
                assertEquals("offset " + read.position() + " is block aligned", 0, read.position() % block);
                assertEquals("length " + read.length() + " is block aligned", 0, read.length() % block);
            }
        }
    }

    public void testCheckIntegrityPassesThroughADirectIOCloneAndThroughAnOffClone() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int beforeIntegrity = channels.reads();
            CodecUtil.checksumEntireFile(fixture.file.clone());
            assertTrue("checkIntegrity read with O_DIRECT", channels.reads() > beforeIntegrity);
            on.set(false);
            final int readsBefore = channels.reads();
            CodecUtil.checksumEntireFile(fixture.file.clone());
            assertEquals("with the index off the integrity read is the original input's", readsBefore, channels.reads());
        }
    }

    public void testTheFooterReadsTheSameWithDirectIOOnAndOff() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final IndexInput directIOClone = fixture.file.clone();
            final long directIOChecksum = CodecUtil.retrieveChecksum(directIOClone);
            on.set(false);
            final IndexInput offClone = fixture.file.clone();
            assertEquals(directIOChecksum, CodecUtil.retrieveChecksum(offClone));
        }
    }

    public void testAShortChannelReadFallsBackToTheOriginalInput() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromDirectIO = new byte[8192];
            final byte[] fromReference = new byte[8192];
            channels.behave(Behaviour.SHORT);
            fixture.directIO.seek(1024);
            fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            fixture.reference.seek(1024);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals("a short read is served by the original input", fromReference, fromDirectIO);
            assertEquals(1024 + fromDirectIO.length, fixture.directIO.getFilePointer());
            // The channel returned bytes rather than throwing, and fewer than the offset asked for, which is the
            // branch this case exists to cover: one device read was attempted, under one block, and the file then
            // reads through the original input.
            assertEquals("one device read was attempted", fixture.readsAtInitialize + 1, channels.reads());
            assertTrue(DirectIOTestUtils.TestChannel.SHORT_READ_BYTES < fixture.blockSize());
            fixture.directIO.seek(0);
            fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            assertEquals("a file whose channel failed never touches it again", fixture.readsAtInitialize + 1, channels.reads());
        }
    }

    public void testAChannelThatThrowsFallsBackToTheOriginalInput() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromDirectIO = new byte[4096];
            final byte[] fromReference = new byte[4096];
            channels.behave(Behaviour.THROW);
            fixture.directIO.seek(2048);
            fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            fixture.reference.seek(2048);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals(fromReference, fromDirectIO);
            assertEquals("one device read was attempted and then the file fell back", fixture.readsAtInitialize + 1, channels.reads());
        }
    }

    public void testTheFallbackOverwritesBytesAPartialReadAlreadyCopied() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromDirectIO = new byte[8192];
            final byte[] fromReference = new byte[8192];
            channels.behave(Behaviour.GARBAGE_THEN_THROW);
            fixture.directIO.seek(0);
            fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            fixture.reference.seek(0);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals("the whole request is read again from its start", fromReference, fromDirectIO);
        }
    }

    public void testTheFailedMarkIsSharedByEverySliceAndClone() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] target = new byte[2048];
            channels.behave(Behaviour.THROW);
            fixture.directIO.seek(0);
            fixture.directIO.readBytes(target, 0, target.length);
            final int readsAfterFailure = channels.last().reads().size();
            channels.behave(Behaviour.NORMAL);

            final IndexInput anotherClone = fixture.file.clone();
            final IndexInput anotherSlice = fixture.file.slice("after", 0, fixture.file.length());
            anotherClone.seek(0);
            anotherClone.readBytes(target, 0, target.length);
            anotherSlice.seek(0);
            anotherSlice.readBytes(target, 0, target.length);
            fixture.directIO.seek(4096);
            fixture.directIO.readBytes(target, 0, target.length);

            assertEquals("no slice or clone of a failed file touches the channel again", readsAfterFailure, channels.last().reads().size());
        }
    }

    public void testAnInterruptedReaderFallsBackAndTheChannelStaysFailed() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromDirectIO = new byte[4096];
            final byte[] fromReference = new byte[4096];
            fixture.reference.seek(0);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            try {
                Thread.currentThread().interrupt();
                fixture.directIO.seek(0);
                fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            } finally {
                Thread.interrupted();
            }
            assertArrayEquals("an interrupted read is served by the original input", fromReference, fromDirectIO);
            final int readsAfterFailure = channels.last().reads().size();
            fixture.directIO.seek(8192);
            fixture.directIO.readBytes(fromDirectIO, 0, fromDirectIO.length);
            assertEquals("the channel an interrupt closed never reopens", readsAfterFailure, channels.last().reads().size());
        }
    }

    public void testTheIntegerReadsFallBackAsWholeRequests() throws IOException {
        try (Fixture fixture = new Fixture()) {
            channels.behave(Behaviour.THROW);
            fixture.directIO.seek(64);
            fixture.reference.seek(64);
            assertEquals(fixture.reference.readLong(), fixture.directIO.readLong());
            assertEquals("one readLong of eight bytes advances eight, not one per fallback", 72L, fixture.directIO.getFilePointer());

            assertEquals(fixture.reference.readInt(), fixture.directIO.readInt());
            assertEquals(76L, fixture.directIO.getFilePointer());
            assertEquals(fixture.reference.readShort(), fixture.directIO.readShort());
            assertEquals(78L, fixture.directIO.getFilePointer());

            final int[] ints = new int[16];
            final int[] referenceInts = new int[16];
            fixture.directIO.readInts(ints, 0, ints.length);
            fixture.reference.readInts(referenceInts, 0, referenceInts.length);
            assertArrayEquals(referenceInts, ints);
            assertEquals("one readInts of sixteen is one request read whole", 78L + 64L, fixture.directIO.getFilePointer());

            final long[] longs = new long[8];
            final long[] referenceLongs = new long[8];
            fixture.directIO.readLongs(longs, 0, longs.length);
            fixture.reference.readLongs(referenceLongs, 0, referenceLongs.length);
            assertArrayEquals(referenceLongs, longs);
            assertEquals(78L + 64L + 64L, fixture.directIO.getFilePointer());
            assertEquals(
                "every one of those requests went to the original input after the first failure",
                fixture.readsAtInitialize + 1,
                channels.reads()
            );
        }
    }

    public void testEveryReadFallsBackWhenItIsTheFirstToFail() throws IOException {
        try (Fixture fixture = new Fixture()) {
            for (final NamedRead read : DirectIOTestUtils.EVERY_READ) {
                channels.behave(Behaviour.NORMAL);
                // The failed mark belongs to one open of the file, so each read gets an open of its own.
                try (IndexInput file = fixture.directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    final IndexInput directIO = file.clone();
                    assertTrue(directIO instanceof DirectIOVectorIndexInput);
                    final int readsBefore = channels.reads();
                    channels.behave(Behaviour.THROW);
                    DirectIOTestUtils.assertReadsAlike(read, fixture.reference, directIO);
                    assertEquals(read.name() + " tried the device once, then read the original input", readsBefore + 1, channels.reads());
                    DirectIOTestUtils.assertReadsAlike(read, fixture.reference, directIO);
                    assertEquals(read.name() + " on a failed file never tries the device again", readsBefore + 1, channels.reads());
                }
            }
        }
    }

    public void testReadsIntsAndFloatsThatStraddleTheEndOfTheBuffer() throws IOException {
        try (Fixture fixture = new Fixture()) {
            // A slice whose offset is not a multiple of four, which is the shape Lucene's flat reader produces: the
            // vector data starts just after the codec header. Every buffer window then ends with one, two or three
            // bytes of a value, which is the hand-rolled reassembly path.
            final long sliceOffset = 3;
            final IndexInput slice = fixture.file.slice("unaligned", sliceOffset, fixture.file.length() - sliceOffset);
            assertTrue(slice instanceof DirectIOVectorIndexInput);
            final int block = fixture.blockSize();
            final int count = 2 * block;
            final float[] fromDirectIO = new float[count];
            final float[] fromReference = new float[count];
            final int[] ints = new int[count];
            final int[] referenceInts = new int[count];
            for (final long start : new long[] { 0, block - 7, 2L * block - 1, 4L * block - 3 }) {
                slice.seek(start);
                slice.readFloats(fromDirectIO, 0, count);
                fixture.reference.seek(sliceOffset + start);
                fixture.reference.readFloats(fromReference, 0, count);
                assertArrayEquals("readFloats at " + start + " of an unaligned slice", fromReference, fromDirectIO, 0.0f);
                slice.seek(start);
                slice.readInts(ints, 0, count);
                fixture.reference.seek(sliceOffset + start);
                fixture.reference.readInts(referenceInts, 0, count);
                assertArrayEquals("readInts at " + start + " of an unaligned slice", referenceInts, ints);
            }
            assertTrue("the straddling reads went to the device and never fell back", channels.reads() > fixture.readsAtInitialize + 4);
        }
    }

    public void testHalfFloatVectorsReadEqualWithDirectIOOnAndOff() throws IOException {
        // A half_float field reads one vector as 2 x dimension bytes through readBytes on a slice, rather than
        // through readFloats, and 1,023 dimensions straddle every block boundary.
        final int dimension = 1_023;
        final int vectorBytes = dimension * Short.BYTES;
        final int vectors = 64;
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, DirectIOTestUtils.HALF_FLOAT_VEC_NAME, vectorBytes * vectors);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (
                IndexInput file = directory.openInput(DirectIOTestUtils.HALF_FLOAT_VEC_NAME, IOContext.DEFAULT);
                IndexInput reference = in.openInput(DirectIOTestUtils.HALF_FLOAT_VEC_NAME, IOContext.DEFAULT)
            ) {
                final IndexInput directIOSlice = file.slice("KNN1040HalfFloatFlatVectorsValuesSlice", 0, (long) vectorBytes * vectors);
                assertTrue(directIOSlice instanceof DirectIOVectorIndexInput);
                final int afterInitialize = channels.reads();
                final byte[] fromDirectIO = new byte[vectorBytes];
                final byte[] fromReference = new byte[vectorBytes];
                for (int ord = 0; ord < vectors; ord++) {
                    directIOSlice.seek((long) ord * vectorBytes);
                    directIOSlice.readBytes(fromDirectIO, 0, vectorBytes);
                    reference.seek((long) ord * vectorBytes);
                    reference.readBytes(fromReference, 0, vectorBytes);
                    assertArrayEquals("vector " + ord, fromReference, fromDirectIO);
                }
                assertTrue("the FP16 reads went to the device", channels.reads() > afterInitialize);

                on.set(false);
                final IndexInput offSlice = file.slice("off", 0, (long) vectorBytes * vectors);
                assertFalse(offSlice instanceof DirectIOVectorIndexInput);
                for (int ord = 0; ord < vectors; ord++) {
                    offSlice.seek((long) ord * vectorBytes);
                    offSlice.readBytes(fromDirectIO, 0, vectorBytes);
                    reference.seek((long) ord * vectorBytes);
                    reference.readBytes(fromReference, 0, vectorBytes);
                    assertArrayEquals("vector " + ord + " with the index off", fromReference, fromDirectIO);
                }
            }
        }
    }
}
