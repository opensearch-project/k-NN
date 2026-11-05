/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ShardPathUtilTests extends KNNTestCase {

    private static final String INDEX_UUID = "V4qM8m9ZTAyRm0Nn0lOAbw";

    public void testTryGetIndexUuidReadsBothDataPathLayouts() throws IOException {
        final Path root = createTempDir();
        final Path defaultLayout = Files.createDirectories(
            root.resolve("nodes").resolve("0").resolve("indices").resolve(INDEX_UUID).resolve("0").resolve("index")
        );
        final Path customLayout = Files.createDirectories(
            root.resolve("shared").resolve("my_data_path").resolve("0").resolve(INDEX_UUID).resolve("0").resolve("index")
        );
        try (Directory in = FSDirectory.open(defaultLayout)) {
            assertEquals(INDEX_UUID, ShardPathUtil.tryGetIndexUuid(in));
            assertEquals("a filter chain is unwrapped", INDEX_UUID, ShardPathUtil.tryGetIndexUuid(new FilterDirectory(in) {
            }));
        }
        try (Directory in = FSDirectory.open(customLayout)) {
            assertEquals("a custom index.data_path has no indices component", INDEX_UUID, ShardPathUtil.tryGetIndexUuid(in));
        }
    }

    public void testTryGetIndexUuidIsNullForAnythingElse() throws IOException {
        final Path root = createTempDir();
        final Path notIndexFolder = Files.createDirectories(root.resolve(INDEX_UUID).resolve("0").resolve("data"));
        final Path nonNumericShard = Files.createDirectories(root.resolve(INDEX_UUID).resolve("shard").resolve("index"));
        final Path tooShallow = Files.createDirectories(root.resolve("index"));
        for (final Path path : new Path[] { notIndexFolder, nonNumericShard, tooShallow }) {
            try (Directory in = FSDirectory.open(path)) {
                assertNull("no UUID for " + path, ShardPathUtil.tryGetIndexUuid(in));
            }
        }
        try (Directory in = new ByteBuffersDirectory()) {
            assertNull("a directory with no folder has no UUID", ShardPathUtil.tryGetIndexUuid(in));
        }
        assertNull(ShardPathUtil.tryGetIndexUuid(null));
    }
}
