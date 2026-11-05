/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.MergeAbortChecker;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class NativeIndexWriterTests extends KNNTestCase {

    public void testMergeIndex_whenWriteFailsWithMergeAbortInCauseChain_thenRethrowsMergeAborted() throws Exception {
        // Mirrors the JNI write path: RateLimitedIndexOutput throws MergeAbortedException, IndexOutputWithBuffer wraps it in a
        // RuntimeException, and the native layer surfaces a generic Exception.
        Exception writeFailure = new Exception(
            "Failed to write index to disk",
            new RuntimeException(new MergePolicy.MergeAbortedException("Merge aborted."))
        );
        try (MockedStatic<MergeAbortChecker> checker = mockStatic(MergeAbortChecker.class)) {
            checker.when(MergeAbortChecker::isMergeAborted).thenReturn(false);
            NativeIndexWriter writer = createMergeWriter(writeFailure);
            expectThrows(MergePolicy.MergeAbortedException.class, () -> writer.mergeIndex(vectorValuesSupplier(), 10));
        }
    }

    public void testMergeIndex_whenWriteFailsAndMergeThreadAborted_thenRethrowsMergeAborted() throws Exception {
        Exception writeFailure = new Exception("Failed to write index to disk");
        try (MockedStatic<MergeAbortChecker> checker = mockStatic(MergeAbortChecker.class)) {
            checker.when(MergeAbortChecker::isMergeAborted).thenReturn(true);
            NativeIndexWriter writer = createMergeWriter(writeFailure);
            expectThrows(MergePolicy.MergeAbortedException.class, () -> writer.mergeIndex(vectorValuesSupplier(), 10));
        }
    }

    public void testMergeIndex_whenWriteFailsWithoutAbort_thenDoesNotThrow() throws Exception {
        Exception writeFailure = new Exception("Failed to write index to disk", new RuntimeException("disk full"));
        try (MockedStatic<MergeAbortChecker> checker = mockStatic(MergeAbortChecker.class)) {
            checker.when(MergeAbortChecker::isMergeAborted).thenReturn(false);
            NativeIndexWriter writer = createMergeWriter(writeFailure);
            writer.mergeIndex(vectorValuesSupplier(), 10);
        }
    }

    private NativeIndexWriter createMergeWriter(Exception writeFailure) throws Exception {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field")
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .addAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            .addAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue())
            .build();
        SegmentWriteState state = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            new ByteBuffersDirectory(),
            mock(SegmentInfo.class),
            mock(FieldInfos.class),
            null,
            IOContext.DEFAULT
        );
        NativeIndexBuildStrategy buildStrategy = mock(NativeIndexBuildStrategy.class);
        doAnswer(invocation -> { throw writeFailure; }).when(buildStrategy).buildAndWriteIndex(any());
        NativeIndexBuildStrategyFactory factory = mock(NativeIndexBuildStrategyFactory.class);
        when(factory.getBuildStrategy(any(), anyInt(), any())).thenReturn(buildStrategy);
        return new NativeIndexWriter(state, fieldInfo, factory, null, null);
    }

    private Supplier<KNNVectorValues<?>> vectorValuesSupplier() {
        KNNVectorValues<?> vectorValues = mock(KNNVectorValues.class);
        when(vectorValues.docId()).thenReturn(0);
        when(vectorValues.bytesPerVector()).thenReturn(16);
        return () -> vectorValues;
    }
}
