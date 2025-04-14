/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.mockito.Mock;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
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
    @Mock
    private NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    private FlatFieldVectorsWriter mockedFlatFieldVectorsWriter;

    private NativeEngines990KnnVectorsWriter objectUnderTest;

    private final String description;
    private final List<Map<Integer, float[]>> vectorsPerField;
    private static final Integer BUILD_GRAPH_ALWAYS_THRESHOLD = 0;
    private static final Integer BUILD_GRAPH_NEVER_THRESHOLD = -1;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
        objectUnderTest = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            BUILD_GRAPH_ALWAYS_THRESHOLD,
            nativeIndexBuildStrategyFactory
        );
        mockedFlatFieldVectorsWriter = Mockito.mock(FlatFieldVectorsWriter.class);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(Mockito.anyInt(), Mockito.any());
        Mockito.when(flatVectorsWriter.addField(Mockito.any())).thenReturn(mockedFlatFieldVectorsWriter);
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
                        Collections.emptyMap(),
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
        final List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = vectorsPerField.stream().map(vectors -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectors.values())
            );
            return KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues);
        }).toList();

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);

                try {
                    objectUnderTest.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
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
                    if (vectorsPerField.get(i).isEmpty()) {
                        verify(nativeIndexWriter, never()).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    } else {
                        verify(nativeIndexWriter).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            final Long expectedTimesGetVectorValuesIsCalled = vectorsPerField.stream().filter(Predicate.not(Map::isEmpty)).count();
            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(Math.toIntExact(expectedTimesGetVectorValuesIsCalled))
            );
        }
    }

    @SneakyThrows
    public void testFlush_WithQuantization() {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));
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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);

                try {
                    objectUnderTest.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
                try {
                    when(
                        quantizationService.train(
                            quantizationParams,
                            expectedVectorValuesSuppliers.get(i),
                            vectorsPerField.get(i).size()
                        )
                    ).thenReturn(quantizationState);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
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
                    if (vectorsPerField.get(i).isEmpty()) {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0), never()).writeState(i, quantizationState);
                        verify(nativeIndexWriter, never()).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    } else {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeState(i, quantizationState);
                        verify(nativeIndexWriter).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            final Long expectedTimesGetVectorValuesIsCalled = vectorsPerField.stream().filter(Predicate.not(Map::isEmpty)).count();
            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(Math.toIntExact(expectedTimesGetVectorValuesIsCalled))
            );
        }
    }

    public void testFlush_whenThresholdIsNegative_thenNativeIndexWriterIsNeverCalled() throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSupplier = new ArrayList<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            expectedVectorValuesSupplier.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));

        });

        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            BUILD_GRAPH_NEVER_THRESHOLD,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);
                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSupplier.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });

            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }
            verifyNoInteractions(nativeIndexWriter);
        }
    }

    public void testFlush_whenThresholdIsGreaterThanVectorSize_thenNativeIndexWriterIsNeverCalled() throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        final Map<Integer, Integer> sizeMap = new HashMap<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            sizeMap.put(i, randomVectorValues.size());
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));

        });
        final int maxThreshold = sizeMap.values().stream().filter(count -> count != 0).max(Integer::compareTo).orElse(0);
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            maxThreshold + 1,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);
                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });

            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }
            verifyNoInteractions(nativeIndexWriter);
        }
    }

    public void testFlush_whenThresholdIsEqualToMinNumberOfVectors_thenNativeIndexWriterIsCalled() throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        final Map<Integer, Integer> sizeMap = new HashMap<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            sizeMap.put(i, randomVectorValues.size());
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));
        });

        final int minThreshold = sizeMap.values().stream().filter(count -> count != 0).min(Integer::compareTo).orElse(0);
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            minThreshold,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);
                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });

            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
                assertTrue((long) KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue() > 0);
            }
            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    if (vectorsPerField.get(i).size() > 0) {
                        verify(nativeIndexWriter).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    } else {
                        verify(nativeIndexWriter, never()).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        }
    }

    public void testFlush_whenThresholdIsEqualToFixedValue_thenRelevantNativeIndexWriterIsCalled() throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));

        });
        final int threshold = 4;
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            threshold,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);
                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });

            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }
            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    if (vectorsPerField.get(i).size() >= threshold) {
                        verify(nativeIndexWriter).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    } else {
                        verify(nativeIndexWriter, never()).flushIndex(expectedVectorValuesSuppliers.get(i), vectorsPerField.get(i).size());
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        }
    }

    public void testFlush_whenQuantizationIsProvided_whenBuildGraphDatStructureThresholdIsNotMet_thenStillBuildGraph() throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        final Map<Integer, Integer> sizeMap = new HashMap<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            sizeMap.put(i, randomVectorValues.size());
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));

        });
        final int maxThreshold = sizeMap.values().stream().filter(count -> count != 0).max(Integer::compareTo).orElse(0);
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            maxThreshold + 1, // to avoid building graph using max doc threshold, the same can be achieved by -1 too,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);

                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
                try {
                    when(
                        quantizationService.train(
                            quantizationParams,
                            expectedVectorValuesSuppliers.get(i),
                            vectorsPerField.get(i).size()
                        )
                    ).thenReturn(quantizationState);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeHeader(segmentWriteState);
            } else {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }
            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    if (vectorsPerField.get(i).isEmpty()) {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0), never()).writeState(i, quantizationState);
                    } else {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeState(i, quantizationState);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            final Long expectedTimesGetVectorValuesIsCalled = vectorsPerField.stream().filter(Predicate.not(Map::isEmpty)).count();
            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(Math.toIntExact(expectedTimesGetVectorValuesIsCalled))
            );
        }
    }

    public void testFlush_whenQuantizationIsProvided_whenBuildGraphDatStructureThresholdIsNegative_thenStillBuildGraph()
        throws IOException {
        // Given
        List<Supplier<KNNVectorValues<?>>> expectedVectorValuesSuppliers = new ArrayList<>();
        final Map<Integer, Integer> sizeMap = new HashMap<>();
        IntStream.range(0, vectorsPerField.size()).forEach(i -> {
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                new ArrayList<>(vectorsPerField.get(i).values())
            );
            sizeMap.put(i, randomVectorValues.size());
            expectedVectorValuesSuppliers.add(KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, randomVectorValues));

        });
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            BUILD_GRAPH_NEVER_THRESHOLD,
            nativeIndexBuildStrategyFactory
        );

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
                fieldWriterMockedStatic.when(
                    () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
                ).thenReturn(field);

                try {
                    nativeEngineWriter.addField(fieldInfo);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
                knnVectorValuesFactoryMockedStatic.when(
                    () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectorsPerField.get(i))
                ).thenReturn(expectedVectorValuesSuppliers.get(i));

                when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
                try {
                    when(
                        quantizationService.train(
                            quantizationParams,
                            expectedVectorValuesSuppliers.get(i),
                            vectorsPerField.get(i).size()
                        )
                    ).thenReturn(quantizationState);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                nativeIndexWriterMockedStatic.when(
                    () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory)
                ).thenReturn(nativeIndexWriter);
            });
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            // When
            nativeEngineWriter.flush(5, null);

            // Then
            verify(flatVectorsWriter).flush(5, null);
            if (vectorsPerField.size() > 0) {
                verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeHeader(segmentWriteState);
            } else {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            }
            IntStream.range(0, vectorsPerField.size()).forEach(i -> {
                try {
                    if (vectorsPerField.get(i).isEmpty()) {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0), never()).writeState(i, quantizationState);
                    } else {
                        verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeState(i, quantizationState);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            final Long expectedTimesGetVectorValuesIsCalled = vectorsPerField.stream().filter(Predicate.not(Map::isEmpty)).count();
            knnVectorValuesFactoryMockedStatic.verify(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(any(VectorDataType.class), any(DocsWithFieldSet.class), any()),
                times(Math.toIntExact(expectedTimesGetVectorValuesIsCalled))
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
        FlatFieldVectorsWriter flatFieldVectorsWriter = mock(FlatFieldVectorsWriter.class);
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        vectors.keySet().stream().sorted().forEach(docsWithFieldSet::add);
        when(fieldVectorsWriter.getFieldInfo()).thenReturn(fieldInfo);
        when(fieldVectorsWriter.getVectors()).thenReturn(vectors);
        when(fieldVectorsWriter.getFlatFieldVectorsWriter()).thenReturn(flatFieldVectorsWriter);
        when(flatFieldVectorsWriter.getDocsWithFieldSet()).thenReturn(docsWithFieldSet);
        return fieldVectorsWriter;
    }
}
