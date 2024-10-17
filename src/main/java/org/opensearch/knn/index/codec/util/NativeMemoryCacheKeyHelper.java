/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.index.SegmentInfo;

import java.util.Base64;

public final class NativeMemoryCacheKeyHelper {
    private NativeMemoryCacheKeyHelper() {}

    /**
     * Construct a unique cache key for look-up operation in {@link org.opensearch.knn.index.memory.NativeMemoryCacheManager}
     *
     * @param vectorIndexFileName Vector index file name. Ex: _0_165_test_field.faiss.
     * @param segmentInfo Segment info object representing a logical segment unit containing a vector index.
     * @return Unique cache key that can be used for look-up and invalidating in
     * {@link org.opensearch.knn.index.memory.NativeMemoryCacheManager}
     */
    public static String constructCacheKey(final String vectorIndexFileName, final SegmentInfo segmentInfo) {
        final String segmentId = Base64.getEncoder().encodeToString(segmentInfo.getId());
        final String cacheKey = vectorIndexFileName + "@" + segmentId;
        return cacheKey;
    }

    public static String extractVectorIndexFileName(final String cacheKey) {
        final int indexOfDelimiter = cacheKey.indexOf('@');
        if (indexOfDelimiter != -1) {
            final String vectorFileName = cacheKey.substring(0, indexOfDelimiter);
            return vectorFileName;
        }
        return null;
    }
}
