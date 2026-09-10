/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.InfoStream;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
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
import java.nio.ByteBuffer;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.ArgumentCaptor;

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
    public void testWriteField_whenSourceIsCbor_thenMasksAndPreservesMediaType() {
        assertVectorMaskedRoundTrip(XContentType.CBOR);
    }

    @SneakyThrows
    public void testWriteField_whenSourceIsSmile_thenMasksAndPreservesMediaType() {
        assertVectorMaskedRoundTrip(XContentType.SMILE);
    }

    @SneakyThrows
    public void testWriteField_whenSourceIsJson_thenMasksAndPreservesMediaType() {
        assertVectorMaskedRoundTrip(XContentType.JSON);
    }

    /**
     * Regression coverage for derived source ingestion of non-JSON (CBOR/SMILE) documents. Before the fix
     * the writer hardcoded a JSON parser when re-parsing the stored {@code _source}, so a CBOR/SMILE
     * {@code _source} threw {@code OpenSearchParseException: "Failed to parse content to map"}. This drives
     * a real {@code _source} in the given media type through {@link KNN10010DerivedSourceStoredFieldsWriter}
     * and asserts (a) no parse exception is thrown, (b) the derived vector field is masked, and (c) the
     * masked bytes handed to the delegate are still encoded in the original media type.
     */
    @SneakyThrows
    private void assertVectorMaskedRoundTrip(MediaType mediaType) {
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
        // Sanity: the field must be derived-enabled so the writer builds a (non-null) vector mask and takes
        // the re-parse path we are exercising.
        assertTrue(IndexUtil.isDerivedEnabledForField(vectorFieldType, mapperService));

        // Encode the _source in the target media type (CBOR/SMILE/JSON).
        Map<String, Object> source = new HashMap<>();
        source.put(fieldName, List.of(1.0f, 2.0f, 3.0f));
        source.put("text_field", "text_value");
        BytesStreamOutput bStream = new BytesStreamOutput();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(mediaType, bStream).map(source);
        builder.close();
        BytesRef sourceBytes = bStream.bytes().toBytesRef();

        KNN10010DerivedSourceStoredFieldsWriter writer = new KNN10010DerivedSourceStoredFieldsWriter(
            "mock-codec",
            delegate,
            segmentInfo,
            mapperService
        );

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(SourceFieldMapper.NAME).build();
        // Must not throw (the regression threw OpenSearchParseException here for CBOR/SMILE).
        writer.writeField(fieldInfo, sourceBytes);

        // Capture the masked bytes handed to the delegate and verify the round-trip.
        ArgumentCaptor<BytesRef> captor = ArgumentCaptor.forClass(BytesRef.class);
        verify(delegate).writeField(same(fieldInfo), captor.capture());
        BytesRef written = captor.getValue();

        // The masked _source must still be readable in the SAME media type it came in as (auto-detected).
        Tuple<? extends MediaType, Map<String, Object>> reparsed = XContentHelper.convertToMap(
            BytesReference.fromByteBuffer(ByteBuffer.wrap(written.bytes, written.offset, written.length)),
            true,
            (MediaType) null
        );
        assertEquals(mediaType, reparsed.v1());
        // The vector field must be masked (replaced with the single-byte MASK), not the original array.
        Object maskedValue = reparsed.v2().get(fieldName);
        assertEquals(KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), ((Number) maskedValue).intValue());
        // Non-vector fields are preserved untouched.
        assertEquals("text_value", reparsed.v2().get("text_field"));
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

    @SneakyThrows
    public void testMerge_whenNonKnnReaderPresent_thenFallsBackToGenericMerge() {
        // When a source reader is not our own reader (e.g. the recovery wrapper introduced by
        // _source.exclude), merge() must NOT take the optimized delegate.merge() block-copy path -
        // it falls back to the generic super.merge(), which re-applies the vector mask per document.
        StoredFieldsWriter delegate = mock(StoredFieldsWriter.class);
        SegmentInfo segmentInfo = mock(SegmentInfo.class);
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldTypes()).thenReturn(Collections.emptyList());

        KNN10010DerivedSourceStoredFieldsWriter writer = new KNN10010DerivedSourceStoredFieldsWriter(
            "mock-codec",
            delegate,
            segmentInfo,
            mapperService
        );

        // A single non-k-NN reader with no documents (maxDoc = 0), so super.merge() has no work to do.
        StoredFieldsReader nonKnnReader = mock(StoredFieldsReader.class);
        MergeState mergeState = mergeStateWithReaders(nonKnnReader);

        int docCount = writer.merge(mergeState);

        assertEquals(0, docCount);
        // Fallback path: the delegate's optimized merge must never be invoked.
        verify(delegate, never()).merge(any());
        // The reader is left in place (not wrapped) because it is not our k-NN reader.
        assertSame(nonKnnReader, mergeState.storedFieldsReaders[0]);
    }

    /**
     * Builds a minimal {@link MergeState} that only populates the fields merge() and the generic
     * super.merge() touch for empty (maxDoc = 0) segments: the readers, per-reader field infos, doc
     * maps, and maxDocs. Everything else is left null.
     */
    private static MergeState mergeStateWithReaders(StoredFieldsReader... readers) {
        int n = readers.length;
        MergeState.DocMap[] docMaps = new MergeState.DocMap[n];
        FieldInfos[] fieldInfos = new FieldInfos[n];
        int[] maxDocs = new int[n];
        for (int i = 0; i < n; i++) {
            docMaps[i] = docId -> docId;
            fieldInfos[i] = new FieldInfos(new FieldInfo[0]);
            maxDocs[i] = 0;
        }
        return new MergeState(
            docMaps,
            null,
            new FieldInfos(new FieldInfo[0]),
            readers,
            null,
            null,
            null,
            fieldInfos,
            null,
            null,
            null,
            null,
            maxDocs,
            InfoStream.NO_OUTPUT,
            Runnable::run,
            false,
            null
        );
    }
}
