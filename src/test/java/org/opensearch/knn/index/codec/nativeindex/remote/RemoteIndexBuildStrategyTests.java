/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.mockito.Mockito;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class RemoteIndexBuildStrategyTests extends OpenSearchTestCase {

    static int fallbackCounter = 0;

    private static class TestIndexBuildStrategy implements NativeIndexBuildStrategy {

        @Override
        public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
            fallbackCounter++;
        }
    }

    public void testFallback() throws IOException {
        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 }, new float[] { 3, 4 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        when(repositoriesService.repository(any())).thenThrow(new RepositoryMissingException("Fallback"));

        RemoteIndexBuildStrategy objectUnderTest = new RemoteIndexBuildStrategy(() -> repositoriesService, new TestIndexBuildStrategy());

        IndexOutputWithBuffer indexOutputWithBuffer = Mockito.mock(IndexOutputWithBuffer.class);

        BuildIndexParams buildIndexParams = BuildIndexParams.builder()
            .indexOutputWithBuffer(indexOutputWithBuffer)
            .knnEngine(KNNEngine.FAISS)
            .vectorDataType(VectorDataType.FLOAT)
            .parameters(Map.of("index", "param"))
            .knnVectorValuesSupplier(() -> knnVectorValues)
            .totalLiveDocs((int) knnVectorValues.totalLiveDocs())
            .build();

        objectUnderTest.buildAndWriteIndex(buildIndexParams);
        assertEquals(1, fallbackCounter);
    }
}
