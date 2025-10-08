/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.mockito.ArgumentCaptor;
import org.opensearch.action.IndicesRequest;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.transport.GetModelAction;
import org.opensearch.knn.plugin.transport.GetModelRequest;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.transport.client.Client;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class MMRTestCase extends KNNTestCase {
    float DELTA = 1e-6F;

    void mockClusterIndexMetadata(final Map<String, Map<String, Object>> indexToMappingMap) {
        ClusterService clusterService = mock(ClusterService.class);
        ClusterState clusterState = mock(ClusterState.class);
        Metadata metadata = mock(Metadata.class);
        when(clusterService.state()).thenReturn(clusterState);
        when(clusterState.metadata()).thenReturn(metadata);

        final Set<String> indices = indexToMappingMap.keySet();
        Map<String, Index> indexNameToIndexMap = new HashMap<>();
        for (String indexName : indices) {
            indexNameToIndexMap.put(indexName, new Index(indexName, "uuid"));
        }
        for (Map.Entry<String, Map<String, Object>> entry : indexToMappingMap.entrySet()) {
            final Index index = indexNameToIndexMap.get(entry.getKey());
            final IndexMetadata indexMetadata = mock(IndexMetadata.class);
            final MappingMetadata mappingMetadata = mock(MappingMetadata.class);
            when(metadata.index(index)).thenReturn(indexMetadata);
            when(indexMetadata.mapping()).thenReturn(mappingMetadata);
            when(indexMetadata.getIndex()).thenReturn(index);
            when(mappingMetadata.sourceAsMap()).thenReturn(entry.getValue());
        }

        IndexNameExpressionResolver resolver = mock(IndexNameExpressionResolver.class);
        // simply return the indices of the request
        when(resolver.concreteIndices(any(ClusterState.class), any(IndicesRequest.class))).thenAnswer(invocation -> {
            IndicesRequest indicesRequest = (IndicesRequest) invocation.getArguments()[1];
            return Arrays.stream(indicesRequest.indices()).map(indexNameToIndexMap::get).toArray(Index[]::new);
        });

        KNNClusterUtil clusterUtil = KNNClusterUtil.instance();
        clusterUtil.initialize(clusterService, resolver);
    }

    void mockModelMetadata(Client mockClient, Map<String, MMRVectorFieldInfo> modelIdToFieldInfoMap) {
        doAnswer(invocation -> {
            GetModelRequest request = (GetModelRequest) invocation.getArguments()[1];
            String modelId = request.getModelID();
            ActionListener<GetModelResponse> getModelListener = invocation.getArgument(2);
            if (modelIdToFieldInfoMap != null && modelIdToFieldInfoMap.containsKey(modelId)) {
                getModelListener.onResponse(createMockGetModelResponse(modelIdToFieldInfoMap.get(modelId)));
            } else {
                getModelListener.onFailure(new Exception("Model ID " + modelId + " not found"));
            }
            return null;
        }).when(mockClient).execute(eq(GetModelAction.INSTANCE), any(GetModelRequest.class), any(ActionListener.class));
    }

    private GetModelResponse createMockGetModelResponse(MMRVectorFieldInfo mmrVectorFieldInfo) {
        GetModelResponse mockResponse = mock(GetModelResponse.class);
        Model mockModel = mock(Model.class);
        ModelMetadata mockModelMetadata = mock(ModelMetadata.class);
        when(mockResponse.getModel()).thenReturn(mockModel);
        when(mockModel.getModelMetadata()).thenReturn(mockModelMetadata);
        when(mockModelMetadata.getSpaceType()).thenReturn(mmrVectorFieldInfo.getSpaceType());
        when(mockModelMetadata.getVectorDataType()).thenReturn(mmrVectorFieldInfo.getVectorDataType());
        return mockResponse;
    }

    <E extends Exception> void verifyException(ActionListener<?> listener, Class<E> expectedType, String expectedMessage) {

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(listener).onFailure(captor.capture());

        Exception exception = captor.getValue();
        assertTrue("Expected " + expectedType.getSimpleName() + " but got " + exception.getClass(), expectedType.isInstance(exception));
        assertEquals(expectedMessage, exception.getMessage());
    }
}
