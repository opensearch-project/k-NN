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
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

public class NativeEngines990KnnVectorsReaderTests extends KNNTestCase {
    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsEnabled_emptyCase() {
        // Prepare field infos
        final FieldInfo[] fieldInfoArray = new FieldInfo[] {};
        final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);

        // Load vector searchers
        final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, Collections.emptySet(), true);
        assertTrue(vectorSearchers.isEmpty());
    }

    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsEnabled_mixedCase() {
        // Prepare field infos
        // - field1: Non KNN field
        // - field2: KNN field, but using Lucene engine
        // - field3: KNN field, FAISS
        // - field4: KNN field, FAISS
        // - field5: KNN field, FAISS, but it does not have file for some reason.

        // Mocking FAISS engine to return a dummy vector searcher
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFactory.createVectorSearcher(any(), any())).thenReturn(mock(VectorSearcher.class));
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            // Prepare field infos
            final FieldInfo[] fieldInfoArray = new FieldInfo[] {
                createFieldInfo("field1", null, 0),
                createFieldInfo("field2", KNNEngine.LUCENE, 1),
                createFieldInfo("field3", mockFaiss, 2),
                createFieldInfo("field4", mockFaiss, 3),
                createFieldInfo("field5", mockFaiss, 4) };
            final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
            final Set<String> filesInSegment = Set.of("_0_165_field3.faiss", "_0_165_field4.faiss");

            // Replace static 'getEngine' to return mockFaiss
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

            // Load vector searchers
            final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, filesInSegment, true);

            // Validate #searchers
            assertEquals(2, vectorSearchers.size());
        }
    }

    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsNotEnabled() {
        // Prepare field infos
        final FieldInfo[] fieldInfoArray = new FieldInfo[] {};
        final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);

        // Load vector searchers
        final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, Collections.emptySet(), false);
        assertNull(vectorSearchers);
    }

    @SneakyThrows
    public void testWhenMemoryOptimizedSearchIsNotEnabled_mixedCase() {
        // Prepare field infos
        // - field1: Non KNN field
        // - field2: KNN field, but using Lucene engine
        // - field3: KNN field, FAISS
        // - field4: KNN field, FAISS
        // - field5: KNN field, FAISS, but it does not have file for some reason.

        // Mocking FAISS engine to return a dummy vector searcher
        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        when(mockFactory.createVectorSearcher(any(), any())).thenReturn(mock(VectorSearcher.class));
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            // Prepare field infos
            final FieldInfo[] fieldInfoArray = new FieldInfo[] {
                createFieldInfo("field1", null, 0),
                createFieldInfo("field2", KNNEngine.LUCENE, 1),
                createFieldInfo("field3", mockFaiss, 2),
                createFieldInfo("field4", mockFaiss, 3),
                createFieldInfo("field5", mockFaiss, 4) };
            final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
            final Set<String> filesInSegment = Set.of("_0_165_field3.faiss", "_0_165_field4.faiss");

            // Replace static 'getEngine' to return mockFaiss
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

            // Load vector searchers
            final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, filesInSegment, false);

            // The table should be null even we had faiss fields.
            assertNull(vectorSearchers);
        }
    }

    private static FieldInfo createFieldInfo(final String fieldName, final KNNEngine engine, final int fieldNo) {
        final KNNCodecTestUtil.FieldInfoBuilder builder = KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName);
        builder.fieldNumber(fieldNo);
        if (engine != null) {
            builder.addAttribute(KNN_FIELD, "true");
            builder.addAttribute(KNN_ENGINE, engine.getName());
        }
        return builder.build();
    }

    @SneakyThrows
    private static Map<String, VectorSearcher> loadSearchers(
        final FieldInfos fieldInfos,
        final Set<String> filesInSegment,
        final boolean memoryOptimizedSearchEnabled
    ) {
        // Prepare infra
        final IndexInput mockIndexInput = mock(IndexInput.class);
        final Directory mockDirectory = mock(Directory.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockIndexInput);
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.files()).thenReturn(filesInSegment);
        when(segmentInfo.getId()).thenReturn((segmentInfo.hashCode() + "").getBytes());
        final SegmentReadState readState = new SegmentReadState(mockDirectory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        // Create reader
        final NativeEngines990KnnVectorsReader reader = new NativeEngines990KnnVectorsReader(readState, null, memoryOptimizedSearchEnabled);
        final Class clazz = NativeEngines990KnnVectorsReader.class;

        // Get searcher table
        final Field tableField = clazz.getDeclaredField("vectorSearchers");
        tableField.setAccessible(true);
        return (Map<String, VectorSearcher>) tableField.get(reader);
    }
}
