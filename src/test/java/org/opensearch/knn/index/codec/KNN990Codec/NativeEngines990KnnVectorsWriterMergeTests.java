/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
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
import java.util.function.Supplier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@RequiredArgsConstructor
public class NativeEngines990KnnVectorsWriterMergeTests extends OpenSearchTestCase {

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
    private FloatVectorValues floatVectorValues;
    @Mock
    private MergeState mergeState;
    @Mock
    private NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    private NativeEngines990KnnVectorsWriter objectUnderTest;

    private final String description;
    private final Map<Integer, float[]> mergedVectors;
    private FlatFieldVectorsWriter mockedFlatFieldVectorsWriter;
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
                $("Merge one field", Map.of(0, new float[] { 1, 2, 3 }, 1, new float[] { 2, 3, 4 }, 2, new float[] { 3, 4, 5 })),
                $("Merge, no live docs", Map.of())
            )
        );
    }

    @SneakyThrows
    public void testMerge() {
        // Given
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(mergedVectors.values())
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = () -> knnVectorValues;

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class);
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mergedVectorValuesMockedStatic = mockStatic(
                KnnVectorsWriter.MergedVectorValues.class
            );
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);
            final FieldInfo fieldInfo = fieldInfo(
                0,
                VectorEncoding.FLOAT32,
                Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
            );

            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, mergedVectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            mergedVectorValuesMockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenReturn(floatVectorValues);
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState)
            ).thenReturn(knnVectorValuesSupplier);

            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
            ).thenReturn(nativeIndexWriter);
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            // When
            objectUnderTest.mergeOneField(fieldInfo, mergeState);

            // Then
            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            if (!mergedVectors.isEmpty()) {
                verify(nativeIndexWriter).mergeIndex(knnVectorValuesSupplier, mergedVectors.size());
                assertTrue(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue() > 0L);
                knnVectorValuesFactoryMockedStatic.verify(
                    () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState),
                    times(1)
                );
            } else {
                verifyNoInteractions(nativeIndexWriter);
            }
        }
    }

    public void testMerge_whenThresholdIsNegative_thenNativeIndexWriterIsNeverCalled() throws IOException {
        // Given
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(mergedVectors.values())
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = () -> knnVectorValues;
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
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mergedVectorValuesMockedStatic = mockStatic(
                KnnVectorsWriter.MergedVectorValues.class
            );
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);
            final FieldInfo fieldInfo = fieldInfo(
                0,
                VectorEncoding.FLOAT32,
                Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
            );

            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, mergedVectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            mergedVectorValuesMockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenReturn(floatVectorValues);
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState)
            ).thenReturn(knnVectorValuesSupplier);

            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
            ).thenReturn(nativeIndexWriter);
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            // When
            nativeEngineWriter.mergeOneField(fieldInfo, mergeState);

            // Then
            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            verifyNoInteractions(nativeIndexWriter);
        }
    }

    public void testMerge_whenThresholdIsEqualToNumberOfVectors_thenNativeIndexWriterIsCalled() throws IOException {
        // Given
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(mergedVectors.values())
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = () -> knnVectorValues;
        final NativeEngines990KnnVectorsWriter nativeEngineWriter = new NativeEngines990KnnVectorsWriter(
            segmentWriteState,
            flatVectorsWriter,
            mergedVectors.size(),
            nativeIndexBuildStrategyFactory
        );
        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class);
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mergedVectorValuesMockedStatic = mockStatic(
                KnnVectorsWriter.MergedVectorValues.class
            );
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);
            final FieldInfo fieldInfo = fieldInfo(
                0,
                VectorEncoding.FLOAT32,
                Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
            );

            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, mergedVectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            mergedVectorValuesMockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenReturn(floatVectorValues);
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState)
            ).thenReturn(knnVectorValuesSupplier);

            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);
            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, null, nativeIndexBuildStrategyFactory)
            ).thenReturn(nativeIndexWriter);
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            // When
            nativeEngineWriter.mergeOneField(fieldInfo, mergeState);

            // Then
            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
            if (!mergedVectors.isEmpty()) {
                verify(nativeIndexWriter).mergeIndex(knnVectorValuesSupplier, mergedVectors.size());
            } else {
                verifyNoInteractions(nativeIndexWriter);
            }
        }
    }

    @SneakyThrows
    public void testMerge_WithQuantization() {
        // Given
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(mergedVectors.values())
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = () -> knnVectorValues;

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class);
            MockedConstruction<KNN990QuantizationStateWriter> knn990QuantWriterMockedConstruction = mockConstruction(
                KNN990QuantizationStateWriter.class
            );
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mergedVectorValuesMockedStatic = mockStatic(
                KnnVectorsWriter.MergedVectorValues.class
            );
        ) {
            quantizationServiceMockedStatic.when(() -> QuantizationService.getInstance()).thenReturn(quantizationService);

            final FieldInfo fieldInfo = fieldInfo(
                0,
                VectorEncoding.FLOAT32,
                Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
            );

            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, mergedVectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            mergedVectorValuesMockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenReturn(floatVectorValues);
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState)
            ).thenReturn(knnVectorValuesSupplier);

            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
            try {
                // Fix mock to use the supplier
                when(quantizationService.train(eq(quantizationParams), any(Supplier.class), eq((long) mergedVectors.size()))).thenReturn(
                    quantizationState
                );
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory)
            ).thenReturn(nativeIndexWriter);
            doAnswer(answer -> {
                Thread.sleep(2); // Need this for KNNGraph value assertion, removing this will fail the assertion
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            // When
            objectUnderTest.mergeOneField(fieldInfo, mergeState);

            // Then
            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            if (!mergedVectors.isEmpty()) {
                verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeHeader(segmentWriteState);
                verify(knn990QuantWriterMockedConstruction.constructed().get(0)).writeState(0, quantizationState);
                verify(nativeIndexWriter).mergeIndex(knnVectorValuesSupplier, mergedVectors.size());
                assertTrue(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue() > 0L);
                knnVectorValuesFactoryMockedStatic.verify(
                    () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(VectorDataType.FLOAT, fieldInfo, mergeState),
                    times(1)
                );
            } else {
                assertEquals(0, knn990QuantWriterMockedConstruction.constructed().size());
                verifyNoInteractions(nativeIndexWriter);
            }
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
