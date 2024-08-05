/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
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

public class MemOptimizedNativeIndexBuildStrategyTests extends OpenSearchTestCase {

    @SneakyThrows
    public void testBuildAndWrite() {
        // Given
        ArgumentCaptor<Long> vectorAddressCaptor = ArgumentCaptor.forClass(Long.class);

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

            // Limits transfer to 2 vectors
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            mockedJNIService.when(() -> JNIService.initIndexFromScratch(3, 2, Map.of("index", "param"), KNNEngine.FAISS)).thenReturn(100L);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .indexPath("indexPath")
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of("index", "param"))
                .build();

            // When
            MemOptimizedNativeIndexBuildStrategy.getInstance().buildAndWriteIndex(buildIndexParams, knnVectorValues);

            mockedJNIService.verify(
                () -> JNIService.initIndexFromScratch(
                    knnVectorValues.totalLiveDocs(),
                    knnVectorValues.dimension(),
                    Map.of("index", "param"),
                    KNNEngine.FAISS
                )
            );

            // Then
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
                () -> JNIService.writeIndex(eq("indexPath"), eq(100L), eq(KNNEngine.FAISS), eq(Map.of("index", "param")))
            );

            assertNotEquals(0L, vectorAddressCaptor.getValue().longValue());
            assertEquals(vectorAddressCaptor.getValue().longValue(), vectorAddressCaptor.getAllValues().get(0).longValue());
        }
    }
}
