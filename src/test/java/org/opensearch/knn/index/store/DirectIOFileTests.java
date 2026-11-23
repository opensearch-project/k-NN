/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The file's own work: the {@code DIRECT} open option and the content check that decides whether a file's bytes read
 * the same through an {@code O_DIRECT} channel and through the directory chain.
 */
public class DirectIOFileTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 64 * 1024;

    private Path shardPath;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
    }

    // No assumption: the shape of the constant is the same on a runtime that cannot open a channel with it, and the
    // branch below is what covers that runtime.
    public void testTheDirectOptionIsLookedUpOncePerJvm() throws Exception {
        final Field field = DirectIOFile.class.getDeclaredField("DIRECT_OPEN_OPTION");
        assertTrue("the option is a constant, not a lookup per call", Modifier.isStatic(field.getModifiers()));
        assertTrue(Modifier.isFinal(field.getModifiers()));
        if (DirectIOTestUtils.isDirectIOSupported()) {
            assertNotNull(DirectIOFile.DIRECT_OPEN_OPTION);
            assertEquals("DIRECT", DirectIOFile.DIRECT_OPEN_OPTION.toString());
        }
    }

    public void testContentMatchesIsFalseWhenTheFileOnDiskIsLonger() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (IndexInput original = in.openInput(VEC_NAME, IOContext.DEFAULT)) {
                // The original input's length is the one it read when it was opened. The file grows underneath it,
                // which is what a layer that pads the bytes it stores looks like from the O_DIRECT side.
                Files.write(shardPath.resolve(VEC_NAME), DirectIOTestUtils.payload(4096), StandardOpenOption.APPEND);
                try (IndexInput grown = in.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    assertDeclinesAgainst(grown, original, "the length the chain reports must be compared against the file on disk");
                }
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheHeaderDiffers() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (Directory crypto = new DirectIOTestUtils.CryptoShapedFsDirectory(shardPath)) {
                try (
                    IndexInput plain = in.openInput(VEC_NAME, IOContext.DEFAULT);
                    IndexInput transformed = crypto.openInput(VEC_NAME, IOContext.DEFAULT)
                ) {
                    assertDeclinesAgainst(plain, transformed, "a layer that transforms the header must be declined");
                }
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheLengthDiffers() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (IndexInput full = in.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput shorter = full.slice("shorter", 0, full.length() - 1);
                assertDeclinesAgainst(full, shorter, "an input one byte shorter than the file must be declined");
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheFooterDiffers() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] bytes = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final String tweaked = "footer-differs.vec";
            final byte[] copy = bytes.clone();
            copy[copy.length - 1] = (byte) (copy[copy.length - 1] ^ 0xFF);
            Files.write(shardPath.resolve(tweaked), copy);
            try (
                IndexInput plain = in.openInput(VEC_NAME, IOContext.DEFAULT);
                IndexInput other = in.openInput(tweaked, IOContext.DEFAULT)
            ) {
                assertDeclinesAgainst(plain, other, "the same length and the same first 4 KiB, a different last byte of the checksum");
            }
        }
    }

    public void testContentMatchesIsTrueForAnInteriorOnlyTransform() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] bytes = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (
                IndexInput interior = new InteriorFlippingIndexInput(in.openInput(VEC_NAME, IOContext.DEFAULT), bytes.length / 2);
                DirectIOFile file = open(in.fileLength(VEC_NAME), interior)
            ) {
                assertNotNull("a transform that leaves the header and the footer alone is not detected, and the javadoc says so", file);
            }
        }
    }

    // DirectIOFile.open returns null for a failed O_DIRECT open as well as for a content mismatch, so each mismatch
    // case first proves that the same file opens and verifies through an input that does match.
    private void assertDeclinesAgainst(final IndexInput matching, final IndexInput mismatching, final String message) throws IOException {
        try (DirectIOFile control = open(matching.length(), matching)) {
            assertNotNull("the file must verify through an input that matches it, or the decline below proves nothing", control);
        }
        assertNull(message, open(mismatching.length(), mismatching));
    }

    private DirectIOFile open(final long length, final IndexInput original) {
        return DirectIOFile.open(shardPath.resolve(VEC_NAME), length, original, DirectIOFile.DEFAULT_CHANNEL_FACTORY);
    }

    /** An input that flips one interior byte, which the content check cannot see. */
    private static final class InteriorFlippingIndexInput extends IndexInput {
        private final IndexInput in;
        private final long flipAt;

        private InteriorFlippingIndexInput(final IndexInput in, final long flipAt) {
            super("interiorFlipping(" + in + ")");
            this.in = in;
            this.flipAt = flipAt;
        }

        @Override
        public byte readByte() throws IOException {
            final long position = in.getFilePointer();
            final byte value = in.readByte();
            return position == flipAt ? (byte) (value ^ 0xFF) : value;
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            final long position = in.getFilePointer();
            in.readBytes(b, offset, len);
            if (position <= flipAt && flipAt < position + len) {
                final int index = offset + (int) (flipAt - position);
                b[index] = (byte) (b[index] ^ 0xFF);
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
            return in.slice(sliceDescription, offset, length);
        }

        @Override
        public IndexInput clone() {
            return new InteriorFlippingIndexInput(in.clone(), flipAt);
        }
    }
}
