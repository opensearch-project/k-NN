/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.mockito.MockedStatic;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class MemoryOptimizedSearchWarmupTests extends KNNTestCase {

    private static final String INDEX_NAME = "test-index";
    private static final String FIELD_MEM_OPT = "mem_opt_field";
    private static final String FIELD_REGULAR = "regular_field";
    private static final String FIELD_NON_KNN = "non_knn_field";

    public void testWarmUp_nullMapperService_returnsEmptyList() {
        LeafReader leafReader = mock(LeafReader.class);
        MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();

        List<String> result = warmup.warmUp(leafReader, null, INDEX_NAME);

        assertTrue(result.isEmpty());
    }

    public void testWarmUp_noMemoryOptimizedFields_returnsEmptyList() {
        // Setup: one field that is NOT a KNN field
        FieldInfo nonKnnField = createFieldInfo(FIELD_NON_KNN, Collections.emptyMap());
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { nonKnnField });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        MapperService mapperService = mock(MapperService.class);

        try (MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class)) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertTrue(result.isEmpty());
        }
    }

    public void testWarmUp_knnFieldNotMemoryOptimized_returnsEmptyList() {
        // Setup: one KNN field that is NOT memory-optimized
        Map<String, String> attrs = new HashMap<>();
        attrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        FieldInfo regularKnnField = createFieldInfo(FIELD_REGULAR, attrs);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { regularKnnField });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        MapperService mapperService = mock(MapperService.class);

        // Field type is KNNVectorFieldType but not supported for memory-optimized search
        KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(mapperService.fieldType(FIELD_REGULAR)).thenReturn(fieldType);

        try (
            MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class);
            MockedStatic<MemoryOptimizedSearchSupportSpec> specMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)
        ) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType, INDEX_NAME)).thenReturn(false);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertTrue(result.isEmpty());
        }
    }

    public void testWarmUp_memoryOptimizedField_warmupSucceeds() throws IOException {
        // Setup: one KNN field that IS memory-optimized
        Map<String, String> attrs = new HashMap<>();
        attrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        FieldInfo memOptField = createFieldInfo(FIELD_MEM_OPT, attrs);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { memOptField });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        WarmableKnnVectorsReader warmableReader = mock(WarmableKnnVectorsReader.class);
        PerFieldKnnVectorsFormat.FieldsReader fieldsReader = mock(PerFieldKnnVectorsFormat.FieldsReader.class);
        when(fieldsReader.getFieldReader(FIELD_MEM_OPT)).thenReturn(warmableReader);
        when(segmentReader.getVectorReader()).thenReturn(fieldsReader);
        doNothing().when(warmableReader).warmUp(eq(FIELD_MEM_OPT));

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(mapperService.fieldType(FIELD_MEM_OPT)).thenReturn(fieldType);

        try (
            MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class);
            MockedStatic<MemoryOptimizedSearchSupportSpec> specMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)
        ) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType, INDEX_NAME)).thenReturn(true);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertEquals(1, result.size());
            assertEquals(FIELD_MEM_OPT, result.get(0));
            verify(warmableReader).warmUp(eq(FIELD_MEM_OPT));
        }
    }

    public void testWarmUp_memoryOptimizedField_warmupThrowsException_returnsEmptyList() throws IOException {
        // Setup: one KNN field that IS memory-optimized but warmup throws
        Map<String, String> attrs = new HashMap<>();
        attrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        FieldInfo memOptField = createFieldInfo(FIELD_MEM_OPT, attrs);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { memOptField });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        WarmableKnnVectorsReader warmableReader = mock(WarmableKnnVectorsReader.class);
        PerFieldKnnVectorsFormat.FieldsReader fieldsReader = mock(PerFieldKnnVectorsFormat.FieldsReader.class);
        when(fieldsReader.getFieldReader(FIELD_MEM_OPT)).thenReturn(warmableReader);
        when(segmentReader.getVectorReader()).thenReturn(fieldsReader);
        doThrow(new RuntimeException("warmup failed")).when(warmableReader).warmUp(eq(FIELD_MEM_OPT));

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(mapperService.fieldType(FIELD_MEM_OPT)).thenReturn(fieldType);

        try (
            MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class);
            MockedStatic<MemoryOptimizedSearchSupportSpec> specMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)
        ) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType, INDEX_NAME)).thenReturn(true);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertTrue(result.isEmpty());
        }
    }

    public void testWarmUp_mixedFields_onlyMemoryOptimizedFieldsReturned() throws IOException {
        // Setup: one memory-optimized KNN field, one regular KNN field, one non-KNN field
        Map<String, String> knnAttrs = new HashMap<>();
        knnAttrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");

        FieldInfo memOptField = createFieldInfo(FIELD_MEM_OPT, knnAttrs);
        FieldInfo regularField = createFieldInfo(FIELD_REGULAR, new HashMap<>(knnAttrs));
        FieldInfo nonKnnField = createFieldInfo(FIELD_NON_KNN, Collections.emptyMap());
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { memOptField, regularField, nonKnnField });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        WarmableKnnVectorsReader warmableReader = mock(WarmableKnnVectorsReader.class);
        PerFieldKnnVectorsFormat.FieldsReader fieldsReader = mock(PerFieldKnnVectorsFormat.FieldsReader.class);
        when(fieldsReader.getFieldReader(FIELD_MEM_OPT)).thenReturn(warmableReader);
        when(segmentReader.getVectorReader()).thenReturn(fieldsReader);
        doNothing().when(warmableReader).warmUp(eq(FIELD_MEM_OPT));

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType memOptFieldType = mock(KNNVectorFieldType.class);
        KNNVectorFieldType regularFieldType = mock(KNNVectorFieldType.class);
        when(mapperService.fieldType(FIELD_MEM_OPT)).thenReturn(memOptFieldType);
        when(mapperService.fieldType(FIELD_REGULAR)).thenReturn(regularFieldType);

        try (
            MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class);
            MockedStatic<MemoryOptimizedSearchSupportSpec> specMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)
        ) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(memOptFieldType, INDEX_NAME)).thenReturn(true);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(regularFieldType, INDEX_NAME)).thenReturn(false);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertEquals(1, result.size());
            assertEquals(FIELD_MEM_OPT, result.get(0));
            // Verify warmUp was only called for the memory-optimized field
            verify(warmableReader).warmUp(eq(FIELD_MEM_OPT));
            verify(warmableReader, never()).warmUp(eq(FIELD_REGULAR));
        }
    }

    public void testWarmUp_multipleMemoryOptimizedFields_partialFailure() throws IOException {
        // Setup: two memory-optimized fields, one succeeds and one fails
        String field1 = "field_success";
        String field2 = "field_fail";

        Map<String, String> knnAttrs = new HashMap<>();
        knnAttrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");

        FieldInfo fieldInfo1 = createFieldInfo(field1, new HashMap<>(knnAttrs));
        FieldInfo fieldInfo2 = createFieldInfo(field2, new HashMap<>(knnAttrs));
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo1, fieldInfo2 });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        WarmableKnnVectorsReader warmableReader1 = mock(WarmableKnnVectorsReader.class);
        WarmableKnnVectorsReader warmableReader2 = mock(WarmableKnnVectorsReader.class);
        PerFieldKnnVectorsFormat.FieldsReader fieldsReader = mock(PerFieldKnnVectorsFormat.FieldsReader.class);
        when(fieldsReader.getFieldReader(field1)).thenReturn(warmableReader1);
        when(fieldsReader.getFieldReader(field2)).thenReturn(warmableReader2);
        when(segmentReader.getVectorReader()).thenReturn(fieldsReader);
        doNothing().when(warmableReader1).warmUp(eq(field1));
        doThrow(new RuntimeException("fail")).when(warmableReader2).warmUp(eq(field2));

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType1 = mock(KNNVectorFieldType.class);
        KNNVectorFieldType fieldType2 = mock(KNNVectorFieldType.class);
        when(mapperService.fieldType(field1)).thenReturn(fieldType1);
        when(mapperService.fieldType(field2)).thenReturn(fieldType2);

        try (
            MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class);
            MockedStatic<MemoryOptimizedSearchSupportSpec> specMock = mockStatic(MemoryOptimizedSearchSupportSpec.class)
        ) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType1, INDEX_NAME)).thenReturn(true);
            specMock.when(() -> MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType2, INDEX_NAME)).thenReturn(true);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertEquals(1, result.size());
            assertEquals(field1, result.get(0));
        }
    }

    public void testWarmUp_fieldTypeNotKNNVectorFieldType_notIncluded() {
        // Setup: KNN_FIELD attribute is set but the MappedFieldType is not KNNVectorFieldType
        Map<String, String> attrs = new HashMap<>();
        attrs.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        FieldInfo field = createFieldInfo(FIELD_REGULAR, attrs);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { field });

        LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFieldInfos()).thenReturn(fieldInfos);

        SegmentReader segmentReader = mock(SegmentReader.class);
        MapperService mapperService = mock(MapperService.class);
        // Return a non-KNNVectorFieldType
        when(mapperService.fieldType(FIELD_REGULAR)).thenReturn(mock(org.opensearch.index.mapper.MappedFieldType.class));

        try (MockedStatic<Lucene> luceneMock = mockStatic(Lucene.class)) {
            luceneMock.when(() -> Lucene.segmentReader(leafReader)).thenReturn(segmentReader);

            MemoryOptimizedSearchWarmup warmup = new MemoryOptimizedSearchWarmup();
            List<String> result = warmup.warmUp(leafReader, mapperService, INDEX_NAME);

            assertTrue(result.isEmpty());
        }
    }

    private static int fieldCounter = 0;

    /**
     * Helper to create a FieldInfo with the given name and attributes.
     */
    private FieldInfo createFieldInfo(String name, Map<String, String> attributes) {
        return new FieldInfo(
            name,
            fieldCounter++,
            false,  // storeTermVector
            false,  // omitNorms
            false,  // storePayloads
            org.apache.lucene.index.IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,     // dvGen
            attributes,
            0,      // pointDimensionCount
            0,      // pointIndexDimensionCount
            0,      // pointNumBytes
            0,      // vectorDimension
            org.apache.lucene.index.VectorEncoding.FLOAT32,
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            false,  // softDeletesField
            false   // parentField
        );
    }

    /**
     * Abstract helper that combines KnnVectorsReader with WarmableReader so
     * Mockito can produce mocks matching the instanceof check in MemoryOptimizedSearchWarmup.
     */
    abstract static class WarmableKnnVectorsReader extends KnnVectorsReader implements WarmableReader {}
}
