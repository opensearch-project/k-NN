/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DerivedSourceVectorTransformerTests extends OpenSearchTestCase {

    private DerivedSourceReaders mockDerivedSourceReaders;
    private SegmentReadState mockSegmentReadState;

    private static final Byte MASK = KNN10010DerivedSourceStoredFieldsWriter.MASK;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockDerivedSourceReaders = mock(DerivedSourceReaders.class);
        mockSegmentReadState = mock(SegmentReadState.class);
    }

    @SuppressWarnings("unchecked")
    public void testInjectVectors_whenNonNested_thenVectorInjected() throws IOException {
        float[] expectedVector = new float[] { 1.5f, 2.5f, 3.5f };
        PerFieldDerivedVectorTransformer mockTransformer = mock(PerFieldDerivedVectorTransformer.class);
        doNothing().when(mockTransformer).setCurrentDoc(anyInt(), anyInt());
        when(mockTransformer.apply(any())).thenReturn(expectedVector);

        try (
            MockedStatic<PerFieldDerivedVectorTransformerFactory> factoryMock = Mockito.mockStatic(
                PerFieldDerivedVectorTransformerFactory.class
            );
            MockedConstruction<DerivedSourceLuceneHelper> helperConstruction = Mockito.mockConstruction(DerivedSourceLuceneHelper.class)
        ) {
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(any(), anyBoolean(), any())).thenReturn(mockTransformer);

            DerivedSourceVectorTransformer transformer = new DerivedSourceVectorTransformer(
                mockDerivedSourceReaders,
                mockSegmentReadState,
                List.of(createMockDerivedFieldInfo("vec", false))
            );

            byte[] source = buildSourceBytes(Map.of("vec", MASK, "text", "hello"));
            byte[] result = transformer.injectVectors(5, source);

            verify(mockTransformer).setCurrentDoc(eq(0), eq(5));
            DerivedSourceLuceneHelper constructedHelper = helperConstruction.constructed().get(0);
            verify(constructedHelper, never()).getFirstChild(anyInt());

            Map<String, Object> resultMap = parseSource(result);
            assertEquals("hello", resultMap.get("text"));
            List<Double> vectorList = (List<Double>) resultMap.get("vec");
            assertEquals(3, vectorList.size());
            assertEquals(1.5, vectorList.get(0), 0.001);
            assertEquals(2.5, vectorList.get(1), 0.001);
            assertEquals(3.5, vectorList.get(2), 0.001);
        }
    }

    @SuppressWarnings("unchecked")
    public void testInjectVectors_whenNested_thenUsesLuceneHelperOffset() throws IOException {
        float[] expectedVector = new float[] { 4.0f, 5.0f };
        PerFieldDerivedVectorTransformer mockTransformer = mock(PerFieldDerivedVectorTransformer.class);
        doNothing().when(mockTransformer).setCurrentDoc(anyInt(), anyInt());
        when(mockTransformer.apply(any())).thenReturn(expectedVector);

        try (
            MockedStatic<PerFieldDerivedVectorTransformerFactory> factoryMock = Mockito.mockStatic(
                PerFieldDerivedVectorTransformerFactory.class
            );
            MockedConstruction<DerivedSourceLuceneHelper> helperConstruction = Mockito.mockConstruction(
                DerivedSourceLuceneHelper.class,
                (mock, context) -> when(mock.getFirstChild(10)).thenReturn(7)
            )
        ) {
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(any(), anyBoolean(), any())).thenReturn(mockTransformer);

            DerivedSourceVectorTransformer transformer = new DerivedSourceVectorTransformer(
                mockDerivedSourceReaders,
                mockSegmentReadState,
                List.of(createMockDerivedFieldInfo("nested_vec", true))
            );

            byte[] source = buildSourceBytes(Map.of("nested_vec", MASK, "id", "doc1"));
            byte[] result = transformer.injectVectors(10, source);

            verify(mockTransformer).setCurrentDoc(eq(7), eq(10));

            Map<String, Object> resultMap = parseSource(result);
            assertEquals("doc1", resultMap.get("id"));
            List<Double> vectorList = (List<Double>) resultMap.get("nested_vec");
            assertEquals(2, vectorList.size());
            assertEquals(4.0, vectorList.get(0), 0.001);
            assertEquals(5.0, vectorList.get(1), 0.001);
        }
    }

    @SuppressWarnings("unchecked")
    public void testInjectVectors_whenMultipleFields_thenAllVectorsInjected() throws IOException {
        float[] vector1 = new float[] { 1.0f, 2.0f };
        float[] vector2 = new float[] { 3.0f, 4.0f };

        PerFieldDerivedVectorTransformer mockTransformer1 = mock(PerFieldDerivedVectorTransformer.class);
        PerFieldDerivedVectorTransformer mockTransformer2 = mock(PerFieldDerivedVectorTransformer.class);
        doNothing().when(mockTransformer1).setCurrentDoc(anyInt(), anyInt());
        doNothing().when(mockTransformer2).setCurrentDoc(anyInt(), anyInt());
        when(mockTransformer1.apply(any())).thenReturn(vector1);
        when(mockTransformer2.apply(any())).thenReturn(vector2);

        DerivedFieldInfo field1 = createMockDerivedFieldInfo("vec_a", false);
        DerivedFieldInfo field2 = createMockDerivedFieldInfo("vec_b", false);

        try (
            MockedStatic<PerFieldDerivedVectorTransformerFactory> factoryMock = Mockito.mockStatic(
                PerFieldDerivedVectorTransformerFactory.class
            );
            MockedConstruction<DerivedSourceLuceneHelper> ignored = Mockito.mockConstruction(DerivedSourceLuceneHelper.class)
        ) {
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(eq(field1.fieldInfo()), eq(false), any()))
                .thenReturn(mockTransformer1);
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(eq(field2.fieldInfo()), eq(false), any()))
                .thenReturn(mockTransformer2);

            DerivedSourceVectorTransformer transformer = new DerivedSourceVectorTransformer(
                mockDerivedSourceReaders,
                mockSegmentReadState,
                List.of(field1, field2)
            );

            byte[] source = buildSourceBytes(Map.of("vec_a", MASK, "vec_b", MASK, "label", "test"));
            byte[] result = transformer.injectVectors(0, source);

            Map<String, Object> resultMap = parseSource(result);
            assertEquals("test", resultMap.get("label"));

            List<Double> resultVec1 = (List<Double>) resultMap.get("vec_a");
            assertEquals(2, resultVec1.size());
            assertEquals(1.0, resultVec1.get(0), 0.001);
            assertEquals(2.0, resultVec1.get(1), 0.001);

            List<Double> resultVec2 = (List<Double>) resultMap.get("vec_b");
            assertEquals(2, resultVec2.size());
            assertEquals(3.0, resultVec2.get(0), 0.001);
            assertEquals(4.0, resultVec2.get(1), 0.001);
        }
    }

    @SuppressWarnings("unchecked")
    public void testInjectVectors_whenMixedNestedAndNonNested_thenAllVectorsInjected() throws IOException {
        float[] nestedVector = new float[] { 10.0f };
        float[] rootVector = new float[] { 20.0f };

        PerFieldDerivedVectorTransformer mockNestedTransformer = mock(PerFieldDerivedVectorTransformer.class);
        PerFieldDerivedVectorTransformer mockRootTransformer = mock(PerFieldDerivedVectorTransformer.class);
        doNothing().when(mockNestedTransformer).setCurrentDoc(anyInt(), anyInt());
        doNothing().when(mockRootTransformer).setCurrentDoc(anyInt(), anyInt());
        when(mockNestedTransformer.apply(any())).thenReturn(nestedVector);
        when(mockRootTransformer.apply(any())).thenReturn(rootVector);

        DerivedFieldInfo nestedField = createMockDerivedFieldInfo("nested_vec", true);
        DerivedFieldInfo rootField = createMockDerivedFieldInfo("root_vec", false);

        try (
            MockedStatic<PerFieldDerivedVectorTransformerFactory> factoryMock = Mockito.mockStatic(
                PerFieldDerivedVectorTransformerFactory.class
            );
            MockedConstruction<DerivedSourceLuceneHelper> helperConstruction = Mockito.mockConstruction(
                DerivedSourceLuceneHelper.class,
                (mock, context) -> when(mock.getFirstChild(5)).thenReturn(3)
            )
        ) {
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(eq(nestedField.fieldInfo()), eq(true), any()))
                .thenReturn(mockNestedTransformer);
            factoryMock.when(() -> PerFieldDerivedVectorTransformerFactory.create(eq(rootField.fieldInfo()), eq(false), any()))
                .thenReturn(mockRootTransformer);

            DerivedSourceVectorTransformer transformer = new DerivedSourceVectorTransformer(
                mockDerivedSourceReaders,
                mockSegmentReadState,
                List.of(nestedField, rootField)
            );

            byte[] source = buildSourceBytes(Map.of("nested_vec", MASK, "root_vec", MASK));
            byte[] result = transformer.injectVectors(5, source);

            // Both get offset=3 from lucene helper since isNested is true
            verify(mockNestedTransformer).setCurrentDoc(eq(3), eq(5));
            verify(mockRootTransformer).setCurrentDoc(eq(3), eq(5));

            Map<String, Object> resultMap = parseSource(result);
            List<Double> nestedResult = (List<Double>) resultMap.get("nested_vec");
            assertEquals(1, nestedResult.size());
            assertEquals(10.0, nestedResult.get(0), 0.001);

            List<Double> rootResult = (List<Double>) resultMap.get("root_vec");
            assertEquals(1, rootResult.size());
            assertEquals(20.0, rootResult.get(0), 0.001);
        }
    }

    private DerivedFieldInfo createMockDerivedFieldInfo(String name, boolean isNested) {
        DerivedFieldInfo mockFieldInfo = mock(DerivedFieldInfo.class);
        when(mockFieldInfo.name()).thenReturn(name);
        when(mockFieldInfo.isNested()).thenReturn(isNested);
        when(mockFieldInfo.fieldInfo()).thenReturn(mock(FieldInfo.class));
        return mockFieldInfo;
    }

    private byte[] buildSourceBytes(Map<String, Object> source) throws IOException {
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(MediaTypeRegistry.getDefaultMediaType());
        builder.map(source);
        builder.close();
        return BytesReference.toBytes(BytesReference.bytes(builder));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseSource(byte[] bytes) {
        return XContentHelper.convertToMap(
            BytesReference.fromByteBuffer(ByteBuffer.wrap(bytes)),
            true,
            MediaTypeRegistry.getDefaultMediaType()
        ).v2();
    }
}
