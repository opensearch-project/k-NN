/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

public class NativeEngines990KnnVectorsReaderTests extends KNNTestCase {
    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsEnabled_emptyCase() {
        // Prepare field infos
        final FieldInfo[] fieldInfoArray = new FieldInfo[] {};
        final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);

        // Load vector searchers
        final NativeEngines990KnnVectorsReader reader = createReader(fieldInfos, Collections.emptySet());

        final NativeEngines990KnnVectorsReader.VectorSearcherHolder vectorSearchers = getVectorSearcherHolders(reader);
        assertFalse(vectorSearchers.isSet());
    }

    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsEnabled_mixedCase() {
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFactory.createVectorSearcher(any(), any(), any(), any())).thenReturn(mock(VectorSearcher.class));
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            // Prepare field infos
            // - field1: Non KNN field
            // - field2: KNN field, but using Lucene engine
            // - field3: KNN field, FAISS
            // - field4: KNN field, FAISS
            // - field5: KNN field, FAISS, but it does not have file for some reason.
            final FieldInfo[] fieldInfoArray = new FieldInfo[] {
                createFieldInfo("field1", null, 0),
                createFieldInfo("field2", KNNEngine.LUCENE, 1),
                createFieldInfo("field3", mockFaiss, 2),
                createFieldInfo("field4", mockFaiss, 3),
                createFieldInfo("field5", mockFaiss, 4) };
            final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
            final Set<String> filesInSegment = Set.of("_0_165_field3.faiss", "_0_165_field4.faiss");

            mockedStatic.when(() -> KNNEngine.getEngine(any())).thenAnswer(invocation -> {
                final String strArg = invocation.getArgument(0);
                // Intercept FAISS engine to return mock
                if (strArg.equals(KNNEngine.FAISS.getName())) {
                    return mockFaiss;
                }

                // Otherwise return Lucene, as field2 is using Lucene.
                return KNNEngine.LUCENE;
            });

            mockedStatic.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

            final NativeEngines990KnnVectorsReader reader_field_2 = createReader(fieldInfos, filesInSegment);
            final NativeEngines990KnnVectorsReader reader_field_3 = createReader(fieldInfos, filesInSegment);
            final NativeEngines990KnnVectorsReader reader_field_4 = createReader(fieldInfos, filesInSegment);

            assertFalse(getVectorSearcherHolders(reader_field_2).isSet());
            assertFalse(getVectorSearcherHolders(reader_field_3).isSet());
            assertFalse(getVectorSearcherHolders(reader_field_4).isSet());

            // Try search for supported field types
            reader_field_2.search("field3", new float[] { 1, 2, 3, 4 }, null, null);
            reader_field_3.search("field4", new float[] { 1, 2, 3, 4 }, null, null);
            Assert.assertThrows(
                UnsupportedOperationException.class,
                () -> { reader_field_4.search("field5", new float[] { 1, 2, 3, 4 }, null, null); }
            );

            // Check holders are set now
            assertTrue(getVectorSearcherHolders(reader_field_2).isSet());
            assertTrue(getVectorSearcherHolders(reader_field_3).isSet());
        }
    }

    private static FieldInfo createFieldInfo(final String fieldName, final KNNEngine engine, final int fieldNo) {
        final KNNCodecTestUtil.FieldInfoBuilder builder = KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName);
        builder.fieldNumber(fieldNo);
        if (engine != null) {
            builder.addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
            builder.addAttribute(KNNConstants.KNN_ENGINE, engine.getName());
        }
        return builder.build();
    }

    @SneakyThrows
    private static NativeEngines990KnnVectorsReader createReader(final FieldInfos fieldInfos, final Set<String> filesInSegment) {
        // Prepare infra
        final IndexInput mockIndexInput = mock(IndexInput.class);
        final Directory mockDirectory = mock(Directory.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockIndexInput);
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.files()).thenReturn(filesInSegment);
        when(segmentInfo.getId()).thenReturn((segmentInfo.hashCode() + "").getBytes());
        final SegmentReadState readState = new SegmentReadState(mockDirectory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        // Create reader
        final NativeEngines990KnnVectorsReader reader = new NativeEngines990KnnVectorsReader(readState, null);
        return reader;
    }

    @SneakyThrows
    private static NativeEngines990KnnVectorsReader.VectorSearcherHolder getVectorSearcherHolders(
        final NativeEngines990KnnVectorsReader reader
    ) {
        // Get searcher table
        final Field tableField = NativeEngines990KnnVectorsReader.class.getDeclaredField("vectorSearcherHolder");
        tableField.setAccessible(true);
        return (NativeEngines990KnnVectorsReader.VectorSearcherHolder) tableField.get(reader);
    }
}
