/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
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
        final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, Collections.emptySet());
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
        final FieldInfo[] fieldInfoArray = new FieldInfo[] {
            createFieldInfo("field1", null, 0),
            createFieldInfo("field2", KNNEngine.LUCENE, 1),
            createFieldInfo("field3", KNNEngine.FAISS, 2),
            createFieldInfo("field4", KNNEngine.FAISS, 3),
            createFieldInfo("field5", KNNEngine.FAISS, 4) };
        final FieldInfos fieldInfos = new FieldInfos(fieldInfoArray);
        final Set<String> filesInSegment = Set.of("_0_165_field3.faiss", "_0_165_field4.faiss");

        // Load vector searchers
        final Map<String, VectorSearcher> vectorSearchers = loadSearchers(fieldInfos, filesInSegment);

        // Validate #searchers
        assertEquals(2, vectorSearchers.size());
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
    private static Map<String, VectorSearcher> loadSearchers(final FieldInfos fieldInfos, final Set<String> filesInSegment) {
        // Prepare infra
        final IndexInput mockIndexInput = mock(IndexInput.class);
        final Directory mockDirectory = mock(Directory.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockIndexInput);
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.files()).thenReturn(filesInSegment);
        when(segmentInfo.getId()).thenReturn((segmentInfo.hashCode() + "").getBytes());
        final SegmentReadState readState = new SegmentReadState(mockDirectory, segmentInfo, fieldInfos, null);

        // Create reader
        final NativeEngines990KnnVectorsReader reader = new NativeEngines990KnnVectorsReader(readState, null);
        final Class clazz = NativeEngines990KnnVectorsReader.class;

        // Call loadMemoryOptimizedSearcher()
        final Method loadMethod = clazz.getDeclaredMethod("loadMemoryOptimizedSearcher");
        loadMethod.setAccessible(true);
        loadMethod.invoke(reader);

        // Get searcher table
        final Field tableField = clazz.getDeclaredField("vectorSearchers");
        tableField.setAccessible(true);
        return (Map<String, VectorSearcher>) tableField.get(reader);
    }
}
