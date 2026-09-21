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
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
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
                .indexParameters(Map.of("index", "param"))
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
                () -> JNIService.writeIndex(
                    eq(indexOutputWithBuffer),
                    eq(100L),
                    eq(KNNEngine.FAISS),
                    eq(Map.of("index", "param")),
                    eq(false)
                )
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

    // --- Coverage: flat storage is never skipped here for any data type - only SQ 1-bit dedupes,
    // and that goes through MemOptimizedScalarQuantizedIndexBuildStrategy instead ---

    @SneakyThrows
    public void testBuildAndWrite_whenHalfFloat_thenDoesNotSkipFlatStorage() {
        assertSkipFlat(VectorDataType.HALF_FLOAT, false);
    }

    @SneakyThrows
    public void testBuildAndWrite_whenFloat_thenDoesNotSkipFlatStorage() {
        assertSkipFlat(VectorDataType.FLOAT, false);
    }

    @SneakyThrows
    public void testBuildAndWrite_whenByte_thenDoesNotSkipFlatStorage() {
        assertSkipFlat(VectorDataType.BYTE, false);
    }

    @SneakyThrows
    private void assertSkipFlat(VectorDataType vectorDataType, boolean expectedSkipFlat) {
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(vectorDataType, 8, 3))
                .thenReturn(offHeapVectorTransfer);
            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(3);
            when(offHeapVectorTransfer.transfer(org.mockito.ArgumentMatchers.any(), eq(false))).thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(true);
            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .field("test_field")
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(vectorDataType)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            mockedJNIService.verify(
                () -> JNIService.writeIndex(
                    eq(indexOutputWithBuffer),
                    eq(100L),
                    eq(KNNEngine.FAISS),
                    eq(Map.of("index", "param")),
                    eq(expectedSkipFlat)
                )
            );
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
                .indexParameters(Map.of("index", "param"))
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
                () -> JNIService.writeIndex(
                    eq(indexOutputWithBuffer),
                    eq(100L),
                    eq(KNNEngine.FAISS),
                    eq(Map.of("index", "param")),
                    eq(false)
                )
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
    public void testBuildAndWrite_freesNativeMemoryOnException() {
        // Given
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            // Simulate exception during vector transfer
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenThrow(new IOException("Chaos Error 5: Input/output error"));

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            RuntimeException exception = expectThrows(
                RuntimeException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - verify free was called to prevent memory leak
            assertTrue(exception.getMessage().contains("Failed to build index"));
            mockedJNIService.verify(() -> JNIService.free(100L, KNNEngine.FAISS));
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_freesNativeMemoryOnIndexBuildAbortedException() {
        // Given
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            // Simulate IndexBuildAbortedException during vector transfer
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenThrow(new IndexBuildAbortedException("Build aborted"));

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            IndexBuildAbortedException abortedException = expectThrows(
                IndexBuildAbortedException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - verify free was called to prevent memory leak even on abort
            assertEquals("Build aborted", abortedException.getMessage());
            mockedJNIService.verify(() -> JNIService.free(100L, KNNEngine.FAISS));
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_doesNotFreeWhenWriteIndexFails() {
        // Given - transfers succeed so control reaches writeIndex, which then throws. Because writeIndex
        // frees the native memory itself (on success and failure), our code must NOT free it again.
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            // Transfers succeed so we reach writeIndex
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(false);
            // writeIndex throws (e.g. I/O error during merge). Native side already freed the allocation.
            mockedJNIService.when(() -> JNIService.writeIndex(Mockito.any(), eq(100L), eq(KNNEngine.FAISS), Mockito.any(), eq(false)))
                .thenThrow(new RuntimeException("Chaos Error 5: Input/output error"));

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            RuntimeException exception = expectThrows(
                RuntimeException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - the original failure propagates and we do NOT free (writeIndex already did)
            assertTrue(exception.getMessage().contains("Failed to build index"));
            mockedJNIService.verify(() -> JNIService.free(Mockito.anyLong(), Mockito.any(KNNEngine.class)), times(0));
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_doesNotFreeOnSuccess() {
        // Given - the happy path completes writeIndex successfully; our code must not free (writeIndex owns it).
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(false);
            // writeIndex succeeds (default mock behaviour is a no-op void)

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams);

            // Then - writeIndex was invoked and we never called free ourselves
            mockedJNIService.verify(() -> JNIService.writeIndex(Mockito.any(), eq(100L), eq(KNNEngine.FAISS), Mockito.any(), eq(false)));
            mockedJNIService.verify(() -> JNIService.free(Mockito.anyLong(), Mockito.any(KNNEngine.class)), times(0));
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_freeFailureDoesNotMaskOriginalException() {
        // Given - transfer fails (so we own cleanup) AND free itself throws. The original failure must be
        // preserved as the thrown exception, with the free failure attached as a suppressed exception.
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenThrow(new IOException("Chaos Error 5: Input/output error"));
            // free throws during cleanup
            mockedJNIService.when(() -> JNIService.free(100L, KNNEngine.FAISS)).thenThrow(new RuntimeException("native free failed"));

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            RuntimeException exception = expectThrows(
                RuntimeException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - original failure is preserved; free failure is attached as suppressed, not masking it
            assertTrue(exception.getMessage().contains("Failed to build index"));
            assertEquals(1, exception.getSuppressed().length);
            assertEquals("native free failed", exception.getSuppressed()[0].getMessage());
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_doesNotFreeWhenInitIndexReturnsZero() {
        // Given - initIndex returns 0 (no valid native allocation) and a failure then occurs before writeIndex.
        // The cleanup must skip freeing to avoid calling free on an invalid (zero) native pointer.
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
            // initIndex returns 0 - no valid allocation
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(0L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            // Force a failure before writeIndex so cleanup runs
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenThrow(new IOException("Chaos Error 5: Input/output error"));

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            RuntimeException exception = expectThrows(
                RuntimeException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - the original failure propagates and free is NOT called on the zero address
            assertTrue(exception.getMessage().contains("Failed to build index"));
            mockedJNIService.verify(() -> JNIService.free(Mockito.anyLong(), Mockito.any(KNNEngine.class)), times(0));
        }
    }

    @SneakyThrows
    public void testBuildAndWrite_doesNotFreeWhenWriteIndexThrowsAbort() {
        // Given - transfers succeed so control reaches writeIndex, which then throws IndexBuildAbortedException.
        // Since writeIndex already owns (and frees) the native memory, our code must NOT free it again.
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
            mockedJNIService.when(() -> JNIService.initIndex(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 8, 3))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getTransferLimit()).thenReturn(2);
            // Transfers succeed so we reach writeIndex
            when(offHeapVectorTransfer.transfer(Mockito.any(), eq(false))).thenReturn(false);
            when(offHeapVectorTransfer.flush(false)).thenReturn(false);
            // writeIndex throws an abort after ownership of the native memory has transferred to it.
            // Use thenAnswer because IndexBuildAbortedException is checked (extends IOException) and is not
            // declared on writeIndex, which Mockito's thenThrow validation would otherwise reject.
            mockedJNIService.when(() -> JNIService.writeIndex(Mockito.any(), eq(100L), eq(KNNEngine.FAISS), Mockito.any(), eq(false)))
                .thenAnswer(invocation -> {
                    throw new IndexBuildAbortedException("Build aborted");
                });

            IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);
            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexOutputWithBuffer(indexOutputWithBuffer)
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .indexParameters(Map.of("index", "param"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
                .build();

            // When
            IndexBuildAbortedException abortedException = expectThrows(
                IndexBuildAbortedException.class,
                () -> MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams)
            );

            // Then - the abort propagates unchanged and we do NOT free (writeIndex already did)
            assertEquals("Build aborted", abortedException.getMessage());
            mockedJNIService.verify(() -> JNIService.free(Mockito.anyLong(), Mockito.any(KNNEngine.class)), times(0));
        }
    }
}
