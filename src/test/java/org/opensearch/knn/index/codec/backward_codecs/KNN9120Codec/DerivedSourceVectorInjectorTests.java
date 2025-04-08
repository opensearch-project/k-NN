/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;

public class DerivedSourceVectorInjectorTests extends KNNTestCase {

    @SneakyThrows
    @SuppressWarnings("unchecked")
    public void testInjectVectors() {
        List<FieldInfo> fields = List.of(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test1").build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test2").build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test3").build()
        );

        Map<String, float[]> fieldToVector = Collections.unmodifiableMap(new HashMap<>() {
            {
                put("test1", new float[] { 1.0f, 2.0f, 3.0f });
                put("test2", new float[] { 4.0f, 5.0f, 6.0f, 7.0f });
                put("test3", new float[] { 7.0f, 8.0f, 9.0f, 1.0f, 3.0f, 4.0f });
                put("test4", null);
            }
        });

        try (MockedStatic<PerFieldDerivedVectorInjectorFactory> factory = Mockito.mockStatic(PerFieldDerivedVectorInjectorFactory.class)) {
            factory.when(() -> PerFieldDerivedVectorInjectorFactory.create(any(), any(), any())).thenAnswer(invocation -> {
                FieldInfo fieldInfo = invocation.getArgument(0);
                return (PerFieldDerivedVectorInjector) (docId, sourceAsMap) -> {
                    float[] vector = fieldToVector.get(fieldInfo.name);
                    if (vector != null) {
                        sourceAsMap.put(fieldInfo.name, vector);
                    }
                };
            });

            DerivedSourceVectorInjector derivedSourceVectorInjector = new DerivedSourceVectorInjector(
                new KNN9120DerivedSourceReaders(null, null, null, null),
                null,
                fields
            );

            int docId = 2;
            String existingFieldKey = "existingField";
            String existingFieldValue = "existingField";
            Map<String, Object> source = Map.of(existingFieldKey, existingFieldValue);
            byte[] originalSourceBytes = mapToBytes(source);
            byte[] modifiedSourceByttes = derivedSourceVectorInjector.injectVectors(docId, originalSourceBytes);
            Map<String, Object> modifiedSource = bytesToMap(modifiedSourceByttes);

            assertEquals(existingFieldValue, modifiedSource.get(existingFieldKey));

            assertArrayEquals(fieldToVector.get("test1"), toFloatArray((List<Double>) modifiedSource.get("test1")), 0.000001f);
            assertArrayEquals(fieldToVector.get("test2"), toFloatArray((List<Double>) modifiedSource.get("test2")), 0.000001f);
            assertArrayEquals(fieldToVector.get("test3"), toFloatArray((List<Double>) modifiedSource.get("test3")), 0.000001f);
            assertFalse(modifiedSource.containsKey("test4"));
        }
    }

    @SneakyThrows
    private byte[] mapToBytes(Map<String, Object> map) {

        BytesStreamOutput bStream = new BytesStreamOutput(1024);
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(MediaTypeRegistry.JSON, bStream).map(map);
        builder.close();
        return BytesReference.toBytes(BytesReference.bytes(builder));
    }

    private float[] toFloatArray(List<Double> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i).floatValue();
        }
        return array;
    }

    private Map<String, Object> bytesToMap(byte[] bytes) {
        Tuple<? extends MediaType, Map<String, Object>> mapTuple = XContentHelper.convertToMap(
            BytesReference.fromByteBuffer(ByteBuffer.wrap(bytes)),
            true,
            MediaTypeRegistry.getDefaultMediaType()
        );

        return mapTuple.v2();
    }

    @SneakyThrows
    public void testShouldInject() {

        List<FieldInfo> fields = List.of(
            KNNCodecTestUtil.FieldInfoBuilder.builder("test1").build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test2").build(),
            KNNCodecTestUtil.FieldInfoBuilder.builder("test3").build()
        );

        DerivedSourceVectorInjector vectorInjector = new DerivedSourceVectorInjector(
            new KNN9120DerivedSourceReaders(null, null, null, null),
            null,
            fields
        );
        assertTrue(vectorInjector.shouldInject(null, null));
        assertTrue(vectorInjector.shouldInject(new String[] { "test1" }, null));
        assertTrue(vectorInjector.shouldInject(new String[] { "test1", "test2", "test3" }, null));
        assertTrue(vectorInjector.shouldInject(null, new String[] { "test2" }));
        assertTrue(vectorInjector.shouldInject(new String[] { "test1" }, new String[] { "test2" }));
        assertTrue(vectorInjector.shouldInject(new String[] { "test1" }, new String[] { "test2", "test3" }));
        assertFalse(vectorInjector.shouldInject(null, new String[] { "test1", "test2", "test3" }));

    }
}
