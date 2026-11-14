/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.common.Nullable;
import org.opensearch.index.shard.ShardPath;

import java.nio.file.Path;

/**
 * Tells which shard a Lucene {@link Directory} belongs to, for code OpenSearch passes no shard context.
 */
@UtilityClass
public class ShardPathUtil {

    /**
     * Returns the UUID of the index whose shard folder a directory reads. A shard's Lucene files live in
     * {@code .../<index uuid>/<shard>/index} under both data path layouts ({@link ShardPath}).
     *
     * @param directory The directory to read. For a segment, pass its own directory ({@code SegmentInfo#dir}): the one
     *                  a codec is handed for a compound segment has no folder.
     * @return The index UUID, or {@code null} when the chain does not end in a file system directory laid out that way.
     */
    @Nullable
    public static String tryGetIndexUuid(@Nullable final Directory directory) {
        if (FilterDirectory.unwrap(directory) instanceof FSDirectory fsDirectory) {
            return indexUuidOf(fsDirectory.getDirectory());
        }
        return null;
    }

    private static String indexUuidOf(final Path indexFolder) {
        if (ShardPath.INDEX_FOLDER_NAME.equals(fileName(indexFolder)) == false) {
            return null;
        }
        final Path shardFolder = indexFolder.getParent();
        if (isShardNumber(fileName(shardFolder)) == false) {
            return null;
        }
        final String uuid = fileName(shardFolder.getParent());
        return uuid == null || uuid.isEmpty() ? null : uuid;
    }

    private static String fileName(@Nullable final Path path) {
        final Path name = path == null ? null : path.getFileName();
        return name == null ? null : name.toString();
    }

    private static boolean isShardNumber(@Nullable final String name) {
        return name != null && name.isEmpty() == false && name.chars().allMatch(Character::isDigit);
    }
}
