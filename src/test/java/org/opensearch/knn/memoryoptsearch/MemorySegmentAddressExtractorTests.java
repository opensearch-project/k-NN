/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@LuceneTestCase.SuppressSysoutChecks(bugUrl = "N/A")
public class MemorySegmentAddressExtractorTests extends LuceneTestCase {
    private static final double MMAP_CHUNK_SIZE_KB = 1024 * 1024;  // 1KB

    @Test
    @SneakyThrows
    public void extractMemorySegmentTest() {
        doExtractMemorySegmentTestWithBaseOffset(7777, 0, 7777, false);
        doExtractMemorySegmentTestWithBaseOffset(7777, 5, 550, false);
        doExtractMemorySegmentTestWithBaseOffset(7777, 5, 3328, false);
        doExtractMemorySegmentTestWithBaseOffset(7777, 2222, 2222, false);
        doExtractMemorySegmentTestWithBaseOffset(7777, 7222, 555, false);
    }

    @Test
    public void exceptionExpectedForInvalidSize() {
        assertThrows(IllegalArgumentException.class, () -> doExtractMemorySegmentTestWithBaseOffset(7777, 0, 7778, true));
        assertThrows(IllegalArgumentException.class, () -> doExtractMemorySegmentTestWithBaseOffset(7777, 0, 10000, true));
        assertThrows(IllegalArgumentException.class, () -> doExtractMemorySegmentTestWithBaseOffset(7777, 5000, 10000, true));
        assertThrows(IllegalArgumentException.class, () -> doExtractMemorySegmentTestWithBaseOffset(7777, 9999, 10000, true));
    }

    @Test
    @SneakyThrows
    public void extractMemorySegmentShouldNullTest() {
        // Create repo
        final Path tempDirPath = createTempDir();

        // Create a dummy file
        final int tmpFileSize = 1333;
        final Path tempFile = Paths.get(tempDirPath.toFile().getAbsolutePath(), "test.bin");
        Files.write(tempFile, new byte[tmpFileSize]);

        // Create directory
        try (final Directory directory = new NIOFSDirectory(tempDirPath)) {
            try (final IndexInput indexInput = directory.openInput(tempFile.getFileName().toString(), IOContext.DEFAULT)) {
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(indexInput, 0, tmpFileSize);

                // Should be null
                assertNull(addressAndSize);
            }
        }
    }

    @SneakyThrows
    private void doExtractMemorySegmentTestWithBaseOffset(
        final int fileSizeBytes,
        final int startOffset,
        final int requestSize,
        final boolean nullExpected
    ) {
        // Create repo
        final Path tempDirPath = createTempDir();

        // Create a dummy file
        final Path tempFile = Paths.get(tempDirPath.toFile().getAbsolutePath(), "test.bin");
        Files.write(tempFile, new byte[fileSizeBytes]);

        // Create directory
        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            try (final IndexInput indexInput = directory.openInput(tempFile.getFileName().toString(), IOContext.DEFAULT)) {
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    indexInput,
                    startOffset,
                    requestSize
                );

                // Should be non-empty array
                assertEquals(nullExpected, addressAndSize == null);
                if (nullExpected == false) {
                    assertTrue(addressAndSize.length > 0);
                    assertEquals(0, addressAndSize.length % 2);
                    assertEquals((int) Math.ceil(requestSize / MMAP_CHUNK_SIZE_KB), addressAndSize.length / 2);

                    // Get size
                    long size = 0;
                    for (int i = 1; i < addressAndSize.length; i += 2) {
                        size += addressAndSize[i];
                    }
                    assertEquals(requestSize, size);
                }
            }
        }
    }
}
