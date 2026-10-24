/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.exception.TerminalIOException;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class NativeIndexWriterTests extends KNNTestCase {
    public void testMergePropagatesTerminalIOException() throws Exception {
        NativeIndexBuildStrategyFactory factory = mock(NativeIndexBuildStrategyFactory.class);
        NativeIndexBuildStrategy strategy = mock(NativeIndexBuildStrategy.class);
        TerminalIOException cleanupFailure = new TerminalIOException("Failed to delete staged remote index file", new IOException());
        when(factory.getBuildStrategy(any(), anyInt(), any())).thenReturn(strategy);
        doThrow(cleanupFailure).when(strategy).buildAndWriteIndex(any());

        FieldInfo field = KNNCodecTestUtil.FieldInfoBuilder.builder("vector")
            .vectorDimension(2)
            .addAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .addAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            .build();
        TestVectorValues.PreDefinedFloatVectorValues values = new TestVectorValues.PreDefinedFloatVectorValues(
            List.of(new float[] { 1, 2 })
        );
        Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, values);

        try (Directory directory = newDirectory()) {
            SegmentInfo segment = KNNCodecTestUtil.newSegmentInfo(directory, "_0", 1, Codec.getDefault());
            SegmentWriteState state = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                directory,
                segment,
                new FieldInfos(new FieldInfo[] { field }),
                null,
                IOContext.DEFAULT
            );
            NativeIndexWriter writer = NativeIndexWriter.getWriter(field, state, null, factory);

            assertSame(cleanupFailure, expectThrows(TerminalIOException.class, () -> writer.mergeIndex(supplier, 1)));
            verify(strategy).buildAndWriteIndex(any());
        }
    }
}
