/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.mockito.Mock;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RequiredArgsConstructor
public class NativeEngines990KnnVectorsWriterFlushTests extends OpenSearchTestCase {

    @Mock
    private FlatVectorsWriter flatVectorsWriter;
    @Mock
    private SegmentWriteState segmentWriteState;
    @Mock
    private QuantizationParams quantizationParams;
    @Mock
    private QuantizationState quantizationState;
    @Mock
    private QuantizationService quantizationService;
    @Mock
    private NativeIndexWriter nativeIndexWriter;

    private NativeEngines990KnnVectorsWriter objectUnderTest;

    private final String description;
    private final List<Map<Integer, float[]>> vectorsPerField;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
        objectUnderTest = new NativeEngines990KnnVectorsWriter(segmentWriteState, flatVectorsWriter);
    }

    @ParametersFactory
    public static Collection<Object[]> data() {
        return Arrays.asList(
            $$(
                $("Single field", List.of(Map.of(0, new float[] { 1, 2, 3 }, 1, new float[] { 2, 3, 4 }, 2, new float[] { 3, 4, 5 }))),
                $("Single field, no total live docs", List.of()),
                $(
                    "Multi Field",
                    List.of(
                        Map.of(0, new float[] { 1, 2, 3 }, 1, new float[] { 2, 3, 4 }, 2, new float[] { 3, 4, 5 }),
                        Map.of(
                            0,
                            new float[] { 1, 2, 3, 4 },
                            1,
                            new float[] { 2, 3, 4, 5 },
                            2,
                            new float[] { 3, 4, 5, 6 },
                            3,
                            new float[] { 4, 5, 6, 7 }
                        )
                    )
                )
            )
        );
    }

    @SneakyThrows
    public void testFlush() {
        // Given
        List<KNNVectorValues<float[]>> expectedVectorValues = new ArrayList<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
                VectorDataType.FLOAT,
                randomVectorValues
            );
            expectedVectorValues.add(knnVectorValues);

        });

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class);
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);
            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                final FieldInfo fieldInfo = fieldInfo(
                    i,
                    VectorEncoding.FLOAT32,
                    Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
                );

                NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, vectorsPerField.get(i));
                fieldWriterMockedStatic.when(() -> NativeEngineFieldVectorsWriter.create(fieldInfo, segmentWriteState.infoStream))
                    .thenReturn(field);

                try {
                    objectUnderTest.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getDocsWithField();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValues.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(() -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null))
                    .thenReturn(nativeIndexWriter);
            });

            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            objectUnderTest.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
                assertNotEquals(0L, (long) KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
            }

            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    verify(nativeIndexWriter).flushIndex(expectedVectorValues.get(i), vectorsPerField.get(i).size());
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValues(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(expectedVectorValues.size())
            );
        }
    }

    @SneakyThrows
    public void testFlush_WithQuantization() {
        // Given
        List<KNNVectorValues<float[]>> expectedVectorValues = new ArrayList<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
                VectorDataType.FLOAT,
                randomVectorValues
            );
            expectedVectorValues.add(knnVectorValues);

        });

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class);
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);

            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                final FieldInfo fieldInfo = fieldInfo(
                    i,
                    VectorEncoding.FLOAT32,
                    Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
                );

                NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, vectorsPerField.get(i));
                fieldWriterMockedStatic.when(() -> NativeEngineFieldVectorsWriter.create(fieldInfo, segmentWriteState.infoStream))
                    .thenReturn(field);

                try {
                    objectUnderTest.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getDocsWithField();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValues.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
                try {
                    when(quantizationService.train(quantizationParams, expectedVectorValues.get(i))).thenReturn(quantizationState);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                nativeIndexWriterMockedStatic.when(() -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState))
                    .thenReturn(nativeIndexWriter);
            });
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            objectUnderTest.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeHeader(segmentWriteState);
                assertTrue(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue() > 0L);
            } else {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }

            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeState(i, quantizationState);
                    verify(nativeIndexWriter).flushIndex(expectedVectorValues.get(i), vectorsPerField.get(i).size());
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });

            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValues(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(expectedVectorValues.size() * 2)
            );
        }
    }

    private FieldInfo fieldInfo(int fieldNumber, VectorEncoding vectorEncoding, Map<String, String> attributes) {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getFieldNumber()).thenReturn(fieldNumber);
        when(fieldInfo.getVectorEncoding()).thenReturn(vectorEncoding);
        when(fieldInfo.attributes()).thenReturn(attributes);
        attributes.forEach((key, value) -> when(fieldInfo.getAttribute(key)).thenReturn(value));
        return fieldInfo;
    }

    private <T> NativeEngineFieldVectorsWriter nativeEngineFieldVectorsWriter(FieldInfo fieldInfo, Map<Integer, T> vectors) {
        NativeEngineFieldVectorsWriter fieldVectorsWriter = mock(NativeEngineFieldVectorsWriter.class);
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        vectors.keySet().stream().sorted().forEach(docsWithFieldSet::add);
        when(fieldVectorsWriter.getFieldInfo()).thenReturn(fieldInfo);
        when(fieldVectorsWriter.getVectors()).thenReturn(vectors);
        when(fieldVectorsWriter.getDocsWithField()).thenReturn(docsWithFieldSet);
        return fieldVectorsWriter;
    }
}
