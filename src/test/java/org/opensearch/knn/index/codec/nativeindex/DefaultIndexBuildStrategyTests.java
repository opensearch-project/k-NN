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
import org.mockito.Mockito;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.mockito.ArgumentMatchers.eq;

public class DefaultIndexBuildStrategyTests extends OpenSearchTestCase {

    private ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);

    @Before
    public void init() {
        vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);
    }

    @SneakyThrows
    public void testBuildAndWrite() {
        // Given
        final Map<Integer, float[]> docs = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 }, 2, new float[] { 3, 4 });
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docs.keySet().stream().sorted().forEach(docsWithFieldSet::add);

        KNNFloatVectorValues knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            docs
        );
        try (
            MockedStatic<KNNSettings> mockedKNNSettings = Mockito.mockStatic(KNNSettings.class);
            MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class)
        ) {

            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            mockedJNIService.when(() -> JNIService.initIndexFromScratch(3, 2, Map.of("index", "param"), KNNEngine.NMSLIB)).thenReturn(100L);

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
                    vectorAddressCaptor.capture(),
                    eq(knnVectorValues.dimension()),
                    eq("indexPath"),
                    eq(Map.of("index", "param")),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            assertNotEquals(0L, vectorAddressCaptor.getValue().longValue());
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
            MockedStatic<KNNSettings> mockedKNNSettings = Mockito.mockStatic(KNNSettings.class);
            MockedStatic<JNIService> mockedJNIService = Mockito.mockStatic(JNIService.class)
        ) {

            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            mockedJNIService.when(
                () -> JNIService.initIndexFromScratch(3, 2, Map.of("model_id", "id", "model_blob", modelBlob), KNNEngine.NMSLIB)
            ).thenReturn(100L);

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
                    vectorAddressCaptor.capture(),
                    eq(2),
                    eq("indexPath"),
                    eq(modelBlob),
                    eq(Map.of("model_id", "id", "model_blob", modelBlob)),
                    eq(KNNEngine.NMSLIB)
                )
            );
            mockedJNIService.verifyNoMoreInteractions();
            assertNotEquals(0L, vectorAddressCaptor.getValue().longValue());
        }
    }
}
