/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.ByteScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.test.OpenSearchTestCase;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class MemOptimizedNativeIndexBuildStrategyTests extends OpenSearchTestCase {

    @SneakyThrows
    public void testBuildAndWrite() {
        // Given
        ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);
        ArgumentCaptor<float[]> vectorTransferCapture = ArgumentCaptor.forClass(float[].class);

        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = Mockito.mockStatic(
                OffHeapVectorTransferFactory.class
            )
        ) {
            // Limits transfer to 2 vectors
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);
            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            when(offHeapVectorTransfer.transfer(vectorTransferCapture.capture(), eq(false))).thenReturn(false)
                .thenReturn(true)
                .thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(true);
            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.initIndex(
                    knnVectorValues.totalLiveDocs(),
                    knnVectorValues.dimension(),
                    Map.of("index", "param"),
                    KNNEngine.FAISS
                )
            );

            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 0, 1 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            // For the flush
            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 2 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            mockedJNIService.verify(
                () -> JNIService.writeIndex(eq(indexOutputWithBuffer), eq(100L), eq(KNNEngine.FAISS), eq(Map.of("index", "param")))
            );
            assertEquals(200L, vectorAddressCaptor.getValue().longValue());
            assertEquals(vectorAddressCaptor.getValue().longValue(), vectorAddressCaptor.getAllValues().get(0).longValue());
            verify(offHeapVectorTransfer, times(0)).reset();

            float[] prev = null;
            for (float[] vector : vectorTransferCapture.getAllValues()) {
                if (prev != null) {
                    assertNotSame(prev, vector);
                }
                prev = vector;
            }
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_withQuantization() {
        // Given
        ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);
        ArgumentCaptor<Object> vectorTransferCapture = ArgumentCaptor.forClass(Object.class);

        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = Mockito.mockStatic(
                OffHeapVectorTransferFactory.class
            );
            MockedStatic<QuantizationService> mockedQuantizationIntegration = Mockito.mockStatic(QuantizationService.class)
        ) {

            // Limits transfer to 2 vectors
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            QuantizationService quantizationService = mock(QuantizationService.class);
            mockedQuantizationIntegration.when(QuantizationService::getInstance).thenReturn(quantizationService);

            QuantizationState quantizationState = mock(QuantizationState.class);
            ArgumentCaptor<float[]> vectorCaptor = ArgumentCaptor.forClass(float[].class);
            // New: Create QuantizationOutput and mock the quantization process
            QuantizationOutput<byte[]> quantizationOutput = mock(QuantizationOutput.class);
            when(quantizationOutput.getQuantizedVectorCopy()).thenReturn(new byte[] { 1, 2 });
            when(quantizationService.createQuantizationOutput(eq(quantizationState.getQuantizationParams()))).thenReturn(
                quantizationOutput
            );

            // Quantize the vector with the quantization output
            when(quantizationService.quantize(eq(quantizationState), vectorCaptor.capture(), eq(quantizationOutput))).thenAnswer(
                invocation -> {
                    quantizationOutput.getQuantizedVectorCopy();
                    return quantizationOutput.getQuantizedVectorCopy();
                }
            );
            when(quantizationState.getDimensions()).thenReturn(2);
            when(quantizationState.getBytesPerVector()).thenReturn(8);

            when(offHeapVectorTransfer.transfer(vectorTransferCapture.capture(), eq(false))).thenReturn(false)
                .thenReturn(true)
                .thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(true);
            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("index", "param"))
                .quantizationState(quantizationState)
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.initIndex(
                    knnVectorValues.totalLiveDocs(),
                    knnVectorValues.dimension(),
                    Map.of("index", "param"),
                    KNNEngine.FAISS
                )
            );

            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 0, 1 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            // For the flush
            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 2 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            mockedJNIService.verify(
                () -> JNIService.writeIndex(eq(indexOutputWithBuffer), eq(100L), eq(KNNEngine.FAISS), eq(Map.of("index", "param")))
            );
            assertEquals(200L, vectorAddressCaptor.getValue().longValue());
            assertEquals(vectorAddressCaptor.getValue().longValue(), vectorAddressCaptor.getAllValues().get(0).longValue());
            verify(offHeapVectorTransfer, times(0)).reset();

            for (Object vector : vectorTransferCapture.getAllValues()) {
                // Assert that the vector is in byte[] format due to quantization
                assertTrue(vector instanceof byte[]);
            }
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_withByteScalarQuantization() {
        // Given
        ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);
        ArgumentCaptor<Object> vectorTransferCapture = ArgumentCaptor.forClass(Object.class);

        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = Mockito.mockStatic(
                OffHeapVectorTransferFactory.class
            );
            MockedStatic<QuantizationService> mockedQuantizationIntegration = Mockito.mockStatic(QuantizationService.class)
        ) {
            byte[] indexTemplate = new byte[] { 1 };

            // Limits transfer to 2 vectors
            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            QuantizationService quantizationService = mock(QuantizationService.class);
            mockedQuantizationIntegration.when(QuantizationService::getInstance).thenReturn(quantizationService);

            ByteScalarQuantizationState quantizationState = mock(ByteScalarQuantizationState.class);
            BuildIndexParams indexInfo = mock(BuildIndexParams.class);
            when(indexInfo.getQuantizationState()).thenReturn(quantizationState);
            when(quantizationState.getIndexTemplate()).thenReturn(indexTemplate);

            mockedJNIService.when(() -> JNIService.initIndexFromTemplate(3, 2, Map.of("index", "param"), KNNEngine.FAISS, indexTemplate))
                .thenReturn(100L);

            when(offHeapVectorTransfer.transfer(vectorTransferCapture.capture(), eq(false))).thenReturn(false)
                .thenReturn(true)
                .thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(true);
            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("index", "param"))
                .quantizationState(quantizationState)
                .vectorValues(knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then
            mockedJNIService.verify(
                () -> JNIService.initIndexFromTemplate(
                    eq(knnVectorValues.totalLiveDocs()),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(KNNEngine.FAISS),
                    eq(indexTemplate)
                )
            );

            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 0, 1 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            // For the flush
            mockedJNIService.verify(
                () -> JNIService.insertToIndex(
                    eq(new int[] { 2 }),
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq(Map.of("index", "param")),
                    eq(100L),
                    eq(KNNEngine.FAISS)
                )
            );

            mockedJNIService.verify(
                () -> JNIService.writeIndex(eq(indexOutputWithBuffer), eq(100L), eq(KNNEngine.FAISS), eq(Map.of("index", "param")))
            );
            assertEquals(200L, vectorAddressCaptor.getValue().longValue());
            assertEquals(vectorAddressCaptor.getValue().longValue(), vectorAddressCaptor.getAllValues().get(0).longValue());
            verify(offHeapVectorTransfer, times(0)).reset();

            for (Object vector : vectorTransferCapture.getAllValues()) {
                assertTrue(vector instanceof float[]);
            }
        }
    }
}
