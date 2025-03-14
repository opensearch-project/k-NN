/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;

public class DerivedSourceStoredFieldsWriterTests extends KNNTestCase {

    @SneakyThrows
    public void testWriteField() {
        StoredFieldsWriter delegate = mock(StoredFieldsWriter.class);
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("_source").build();
        List<String> fields = List.of("test");

        KNN10010DerivedSourceStoredFieldsWriter derivedSourceStoredFieldsWriter = new KNN10010DerivedSourceStoredFieldsWriter(
            delegate,
            fields
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
