/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.test.OpenSearchTestCase;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class DefaultIndexBuildStrategyTests extends OpenSearchTestCase {

    ArgumentCaptor<float[]> vectorTransferCapture = ArgumentCaptor.forClass(float[].class);

    @Before
    public void init() {
        vectorTransferCapture = ArgumentCaptor.forClass(float[].class);
    }

    @SneakyThrows
    public void testBuildAndWrite() {
        // Given
        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });

        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        try (
            MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class);
            MockedStatic<JNIService> mockedJNIService = mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = mockStatic(OffHeapVectorTransferFactory.class)
        ) {

            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 2))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexPath("indexPath")
                .knnEngine(KNNEngine.NMSLIB)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("index", "param"))
                .build();

            // When
            DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams, knnVectorValues);

            // Then
            mockedJNIService.verify(
                () -> JNIService.createIndex(
                    eq(new int[] { 0, 1, 2 }),
                    eq(200L),
                    eq(knnVectorValues.dimension()),
                    eq("indexPath"),
                    eq(Map.of("index", "param")),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            verify(offHeapVectorTransfer).flush(true);
            verify(offHeapVectorTransfer, times(3)).transfer(vectorTransferCapture.capture(), eq(true));

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
    public void testBuildAndWriteWithModel() {
        // Given
        final Map<Integer, float[]> docs = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 }, 2, new float[] { 3, 4 });
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docs.keySet().stream().sorted().forEach(docsWithFieldSet::add);

        byte[] modelBlob = new byte[] { 1 };

        KNNFloatVectorValues knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            docs
        );
        try (
            MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class);
            MockedStatic<JNIService> mockedJNIService = mockStatic(JNIService.class);
            MockedStatic<OffHeapVectorTransferFactory> mockedOffHeapVectorTransferFactory = mockStatic(OffHeapVectorTransferFactory.class)
        ) {

            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            OffHeapVectorTransfer offHeapVectorTransfer = mock(OffHeapVectorTransfer.class);
            mockedOffHeapVectorTransferFactory.when(() -> OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 2))
                .thenReturn(offHeapVectorTransfer);

            when(offHeapVectorTransfer.getVectorAddress()).thenReturn(200L);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexPath("indexPath")
                .knnEngine(KNNEngine.NMSLIB)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("model_id", "id", "model_blob", modelBlob))
                .build();

            // When
            DefaultIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams, knnVectorValues);

            // Then
            mockedJNIService.verify(
                () -> JNIService.createIndexFromTemplate(
                    eq(new int[] { 0, 1, 2 }),
                    eq(200L),
                    eq(2),
                    eq("indexPath"),
                    eq(modelBlob),
                    eq(Map.of("model_id", "id", "model_blob", modelBlob)),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            verify(offHeapVectorTransfer).flush(true);
            verify(offHeapVectorTransfer, times(3)).transfer(vectorTransferCapture.capture(), eq(true));

            float[] prev = null;
            for (float[] vector : vectorTransferCapture.getAllValues()) {
                if (prev != null) {
                    assertNotSame(prev, vector);
                }
                prev = vector;
            }
        }
    }
}
