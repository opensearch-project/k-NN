/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.index.SegmentInfo;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NativeMemoryCacheKeyHelperTests extends KNNTestCase {

    public void test_constructCacheKey_thenSuccess() {
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.getId()).thenReturn(new byte[] { 1, 2, 3, 4 });

        String result = NativeMemoryCacheKeyHelper.constructCacheKey("test.faiss", segmentInfo);

        assertTrue(result.contains("@"));
        assertTrue(result.startsWith("test.faiss@"));
    }

    public void test_extractVectorIndexFileName_thenSuccess() {
        String result = NativeMemoryCacheKeyHelper.extractVectorIndexFileName("_0_165_test_field.faiss@AQIDBA==");

        assertEquals("_0_165_test_field.faiss", result);
    }

    public void test_extractVectorIndexFileNameWithAtInFilename_thenSuccess() {
        String result = NativeMemoryCacheKeyHelper.extractVectorIndexFileName("test@field.faiss@AQIDBA==");

        assertEquals("test@field.faiss", result);
    }

    public void test_extractVectorIndexFileNameInvalidFormat_thenReturnsNull() {
        String result = NativeMemoryCacheKeyHelper.extractVectorIndexFileName("invalidkey");

        assertNull(result);
    }

    public void test_roundTripWithAtInFilename_thenSuccess() {
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.getId()).thenReturn(new byte[] { 1, 2, 3, 4 });
        String originalFileName = "test@field@name.faiss";

        String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(originalFileName, segmentInfo);
        String extractedFileName = NativeMemoryCacheKeyHelper.extractVectorIndexFileName(cacheKey);

        assertEquals(originalFileName, extractedFileName);
    }
}
