/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;

import static org.mockito.Mockito.mockStatic;

public class PrefetchHelperTests extends KNNTestCase {

    public void testPrefetch_whenNullOrds_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(1024);
        PrefetchHelper.prefetch(trackingInput, 0L, 16L, null, 0);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenZeroOrds_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(1024);
        int[] ords = { 0, 1, 2 };
        PrefetchHelper.prefetch(trackingInput, 0L, 16L, ords, 0);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenSingleOrd_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
        int[] ords = { 5 };
        PrefetchHelper.prefetch(trackingInput, 100L, 16L, ords, 1);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenPrefetchDisabled_thenNoPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(false);
            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(0, trackingInput.prefetchCalls.size());
            assertTrue(trackingInput.prefetchCalls.isEmpty());
        }
    }

    public void testPrefetch_whenVectorsWithinSameGroup_thenSingleExactPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(3 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenVectorsSpanMultipleGroups_thenMultipleExactPrefetches() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(500 * 1024);
            int[] ords = { 0, 100, 300 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(101 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(300 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenNumOrdsLessThanArrayLength_thenOnlyPrefetchesNumOrds() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2, 3, 4 };
            PrefetchHelper.prefetch(trackingInput, 0L, 8L, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(16L, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenUnorderedOrds_thenSortsAndGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(500 * 1024);
            int[] ords = { 300, 0, 100 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(101 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(300 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenBaseOffsetNonZero_thenOffsetsCalculatedCorrectly() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            long baseOffset = 10000L;
            int[] ords = { 0, 1 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, baseOffset, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(baseOffset, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(2 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenVectorsExactly128KBApart_thenCreatesNewGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 128 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(128 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenVectorsJustUnder128KBApart_thenSameGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 127 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(128 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenMultipleGroupsAndFileBoundary_thenRespectsFileLength() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            int fileSize = 350 * 1024;
            TrackingIndexInput trackingInput = createTrackingInput(fileSize);
            int[] ords = { 0, 100, 340 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(101 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(340 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenFinalVectorEndsExactlyAtFileLength_thenPrefetchesExactly() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            int fileSize = 100 * 1024;
            TrackingIndexInput trackingInput = createTrackingInput(fileSize);
            int[] ords = { 0, 99 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(100 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(fileSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenGroupLengthExactly128KB_thenPrefetches128KB() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 127 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(128 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenMultipleVectorsAt128KBIntervals_thenMultipleGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(600 * 1024);
            int[] ords = { 0, 128, 256 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(3, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(128 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
            assertEquals(256 * vectorSize, trackingInput.prefetchCalls.get(2).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(2).length());
        }
    }

    public void testPrefetch_whenDuplicateOrds_thenSortsAndDeduplicatesInPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 5, 2, 5, 2, 10 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 5);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(2 * vectorSize, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(9 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenThreeOrMoreGroups_thenCreatesAllGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(800 * 1024);
            int[] ords = { 0, 50, 200, 400, 600 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 5);
            assertEquals(4, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(51 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(200 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
            assertEquals(400 * vectorSize, trackingInput.prefetchCalls.get(2).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(2).length());
            assertEquals(600 * vectorSize, trackingInput.prefetchCalls.get(3).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(3).length());
        }
    }

    public void testPrefetch_whenVectorEndExactly128KBPlusOneByte_thenCreatesNewGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 127 };
            long vectorSize = 1025L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(127 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    private TrackingIndexInput createTrackingInput(int sizeInBytes) throws IOException {
        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("test", IOContext.DEFAULT)) {
            out.writeBytes(new byte[sizeInBytes], sizeInBytes);
        }
        IndexInput delegate = dir.openInput("test", IOContext.DEFAULT);
        return new TrackingIndexInput(delegate);
    }
}
