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
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.util.Collections;
import java.util.Map;

import static org.mockito.Mockito.mock;
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
}
