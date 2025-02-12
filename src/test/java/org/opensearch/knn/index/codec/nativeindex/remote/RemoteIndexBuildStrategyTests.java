/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.remote.RemoteBuildRequest;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;

public class RemoteIndexBuildStrategyTests extends KNNTestCase {

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

    public void testBuildRequest() throws IOException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository blobStoreRepository = mock(BlobStoreRepository.class);
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put("bucket", "test-bucket").build();

        when(metadata.type()).thenReturn("s3");
        when(metadata.settings()).thenReturn(repoSettings);
        when(blobStoreRepository.getMetadata()).thenReturn(metadata);
        when(repositoriesService.repository("test-repo")).thenReturn(blobStoreRepository);

        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        when(knnSettingsMock.getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey())).thenReturn("test-repo");

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            RemoteIndexBuildStrategy objectUnderTest = new RemoteIndexBuildStrategy(
                () -> repositoriesService,
                new TestIndexBuildStrategy()
            );

            List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                vectorValues
            );
            final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
                VectorDataType.FLOAT,
                randomVectorValues
            );

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of(KNNConstants.SPACE_TYPE, "l2"))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs(vectorValues.size())
                .build();

            RemoteBuildRequest request = objectUnderTest.constructBuildRequest(buildIndexParams);

            assertEquals("s3", request.getRepositoryType());
            assertEquals("test-bucket", request.getContainerName());
            assertEquals("faiss", request.getEngine());
            assertEquals("float", request.getDataType()); // TODO this will be in {fp16, fp32, byte, binary}
            assertEquals(vectorValues.size(), request.getDocCount());
        }
    }
}
