/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.MappingLookup;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceSegmentAttributeParser;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.util.IndexUtil;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DerivedSourceStoredFieldsWriterTests extends KNNTestCase {

    @SneakyThrows
    public void testWriteField() {
        // Mock dependencies
        StoredFieldsWriter delegate = mock(StoredFieldsWriter.class);
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        MapperService mapperService = mock(MapperService.class);

        // Mock mapperService to return empty field types (no KNN fields)
        // This means vectorMask will be null and writeField will just delegate
        when(mapperService.fieldTypes()).thenReturn(Collections.emptyList());

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("_source").build();

        KNN10010DerivedSourceStoredFieldsWriter derivedSourceStoredFieldsWriter = new KNN10010DerivedSourceStoredFieldsWriter(
            "mock-codec",
            delegate,
            segmentInfo,
            mapperService
        );

        Map<String, Object> source = Map.of("test", new float[] { 1.0f, 2.0f, 3.0f }, "text_field", "text_value");
        BytesStreamOutput bStream = new BytesStreamOutput();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(MediaTypeRegistry.JSON, bStream).map(source);
        builder.close();
        byte[] originalBytes = bStream.bytes().toBytesRef().bytes;
        byte[] shiftedBytes = new byte[originalBytes.length + 2];
        System.arraycopy(originalBytes, 0, shiftedBytes, 1, originalBytes.length);
        derivedSourceStoredFieldsWriter.writeField(fieldInfo, new BytesRef(shiftedBytes, 1, originalBytes.length));
    }

    @SneakyThrows
    public void testWriteFieldPreservesNonXContentSource() {
        StoredFieldsWriter delegate = mock(StoredFieldsWriter.class);
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        MapperService mapperService = mock(MapperService.class);
        DocumentMapper documentMapper = mock(DocumentMapper.class);
        MappingLookup mappingLookup = mock(MappingLookup.class);
        KNNVectorFieldType vectorFieldType = mock(KNNVectorFieldType.class);

        String fieldName = "vector";
        when(vectorFieldType.name()).thenReturn(fieldName);
        when(mapperService.fieldTypes()).thenReturn(List.of(vectorFieldType));
        when(mapperService.documentMapper()).thenReturn(documentMapper);
        when(documentMapper.metadataMapper(SourceFieldMapper.class)).thenReturn(null);
        when(documentMapper.mappers()).thenReturn(mappingLookup);
        when(mappingLookup.getMapper(fieldName)).thenReturn(null);
        when(mappingLookup.getNestedScope(fieldName)).thenReturn(null);

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(SourceFieldMapper.NAME).build();
        BytesRef rawSource = new BytesRef("filling gaps");
        KNN10010DerivedSourceStoredFieldsWriter derivedSourceStoredFieldsWriter = new KNN10010DerivedSourceStoredFieldsWriter(
            "mock-codec",
            delegate,
            segmentInfo,
            mapperService
        );

        derivedSourceStoredFieldsWriter.writeField(fieldInfo, rawSource);

        verify(delegate).writeField(same(fieldInfo), same(rawSource));
    }

    @SneakyThrows
    public void testFinishPreservesMixedCaseVectorFieldNamesInSegmentAttributes() {
        StoredFieldsWriter delegate = mock(StoredFieldsWriter.class);
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        MapperService mapperService = mock(MapperService.class);
        DocumentMapper documentMapper = mock(DocumentMapper.class);
        MappingLookup mappingLookup = mock(MappingLookup.class);
        KNNVectorFieldType vectorFieldType = mock(KNNVectorFieldType.class);
        Map<String, String> fakeAttributes = new HashMap<>();

        String fieldName = "vectorSearch.nameVector";
        when(vectorFieldType.name()).thenReturn(fieldName);
        when(mapperService.fieldTypes()).thenReturn(List.of(vectorFieldType));
        when(mapperService.documentMapper()).thenReturn(documentMapper);
        when(documentMapper.metadataMapper(SourceFieldMapper.class)).thenReturn(null);
        when(documentMapper.mappers()).thenReturn(mappingLookup);
        when(mappingLookup.getMapper(fieldName)).thenReturn(null);
        when(mappingLookup.getNestedScope(fieldName)).thenReturn(null);
        when(segmentInfo.putAttribute(any(), any())).thenAnswer(t -> fakeAttributes.put(t.getArgument(0), t.getArgument(1)));
        when(segmentInfo.getAttribute(any())).thenAnswer(t -> fakeAttributes.get(t.getArgument(0)));

        assertTrue(IndexUtil.isDerivedEnabledForField(vectorFieldType, mapperService));

        KNN10010DerivedSourceStoredFieldsWriter derivedSourceStoredFieldsWriter = new KNN10010DerivedSourceStoredFieldsWriter(
            "mock-codec",
            delegate,
            segmentInfo,
            mapperService
        );

        derivedSourceStoredFieldsWriter.finish(1);

        assertEquals(List.of(fieldName), DerivedSourceSegmentAttributeParser.parseDerivedVectorFields(segmentInfo, false));
        verify(delegate).finish(1);
    }
}
