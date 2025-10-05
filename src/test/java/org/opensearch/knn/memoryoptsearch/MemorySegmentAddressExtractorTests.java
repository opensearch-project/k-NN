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

public class MemorySegmentAddressExtractorTests extends LuceneTestCase {
    @Test
    @SneakyThrows
    public void extractMemorySegmentTest() {
        doExtractMemorySegmentTestWithBaseOffset(0);
        doExtractMemorySegmentTestWithBaseOffset(555);
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
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(indexInput, 0);

                // Should be null
                assertNull(addressAndSize);
            }
        }
    }

    @SneakyThrows
    private void doExtractMemorySegmentTestWithBaseOffset(int startOffset) {
        // Create repo
        final Path tempDirPath = createTempDir();

        // Create a dummy file
        final int tmpFileSize = startOffset + 1024;
        final Path tempFile = Paths.get(tempDirPath.toFile().getAbsolutePath(), "test.bin");
        Files.write(tempFile, new byte[tmpFileSize]);

        // Create directory
        try (final Directory directory = new MMapDirectory(tempDirPath)) {
            try (final IndexInput indexInput = directory.openInput(tempFile.getFileName().toString(), IOContext.DEFAULT)) {
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(indexInput, startOffset);

                // Should be non-empty array
                assertNotNull(addressAndSize);
                assertTrue(addressAndSize.length > 0);
                assertEquals(0, addressAndSize.length % 2);

                // Get size
                long size = 0;
                for (int i = 1; i < addressAndSize.length; i += 2) {
                    size += addressAndSize[i];
                }
                assertEquals(tmpFileSize - startOffset, size);
            }
        }
    }
}
