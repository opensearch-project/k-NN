/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.training;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

public class TrainingJobClusterStateListenerTests extends KNNTestCase {
    public void testClusterChanged() throws InterruptedException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobClusterStateListener trainingJobClusterStateListener = TrainingJobClusterStateListener.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);
        doAnswer(invocationOnMock -> { return null; }).when(threadPool)
            .schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        ModelDao modelDao = mock(ModelDao.class);
        ClusterChangedEvent clusterChangedEvent = mock(ClusterChangedEvent.class);
        when(clusterChangedEvent.localNodeClusterManager()).thenReturn(true);
        when(clusterChangedEvent.isNewCluster()).thenReturn(true);

        TrainingJobClusterStateListener.initialize(threadPool, modelDao, clusterService);

        trainingJobClusterStateListener.clusterChanged(clusterChangedEvent);

        verify(threadPool, times(1)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        when(clusterChangedEvent.isNewCluster()).thenReturn(false);
        when(clusterChangedEvent.nodesRemoved()).thenReturn(true);
        DiscoveryNodes.Delta delta = mock(DiscoveryNodes.Delta.class);
        List<DiscoveryNode> nodes = new ArrayList<>();
        when(clusterChangedEvent.nodesDelta()).thenReturn(delta);
        when(delta.removedNodes()).thenReturn(nodes);

        trainingJobClusterStateListener.clusterChanged(clusterChangedEvent);

        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));
        verify(clusterChangedEvent, times(1)).nodesDelta();

        when(clusterChangedEvent.nodesRemoved()).thenReturn(false);
        trainingJobClusterStateListener.clusterChanged(clusterChangedEvent);
        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        when(clusterChangedEvent.localNodeClusterManager()).thenReturn(false);
        trainingJobClusterStateListener.clusterChanged(clusterChangedEvent);
        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    }

    public void testUpdateModelsNewCluster() throws IOException, InterruptedException, ExecutionException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobClusterStateListener trainingJobClusterStateListener = TrainingJobClusterStateListener.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getState()).thenReturn(ModelState.TRAINING);
        when(model.getModelMetadata()).thenReturn(modelMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.isCreated()).thenReturn(true);
        when(modelDao.get(modelId)).thenReturn(model);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        doAnswer(invocationOnMock -> {
            SearchResponse searchResponse = mock(SearchResponse.class);
            SearchHits searchHits = mock(SearchHits.class);
            when(searchResponse.getHits()).thenReturn(searchHits);
            SearchHit searchHit = mock(SearchHit.class);
            when(searchHit.getId()).thenReturn(modelId);
            SearchHit[] searchHitArray = new SearchHit[1];
            searchHitArray[0] = searchHit;
            when(searchHits.getHits()).thenReturn(searchHitArray);
            ((ActionListener<SearchResponse>) invocationOnMock.getArguments()[1]).onResponse(searchResponse);
            return null;
        }).when(modelDao).search(any(SearchRequest.class), any(ActionListener.class));
        doAnswer(invocationOnMock -> { return null; }).when(modelDao).update(any(Model.class), any(ActionListener.class));

        TrainingJobClusterStateListener.initialize(threadPool, modelDao, clusterService);

        trainingJobClusterStateListener.updateModelsNewCluster();

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        verify(modelMetadata, times(1)).setState(ModelState.FAILED);
        verify(modelMetadata, times(1)).setError("Training failed to complete as cluster crashed");
        verify(modelDao, times(1)).update(any(Model.class), any(ActionListener.class));
    }

    public void testUpdateModelsNodesRemoved() throws IOException, InterruptedException, ExecutionException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobClusterStateListener trainingJobClusterStateListener = TrainingJobClusterStateListener.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getState()).thenReturn(ModelState.TRAINING);
        when(modelMetadata.getNodeAssignment()).thenReturn("test-node-model-match");
        when(model.getModelMetadata()).thenReturn(modelMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.isCreated()).thenReturn(true);
        when(modelDao.get(modelId)).thenReturn(model);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);
        DiscoveryNode node1 = mock(DiscoveryNode.class);
        when(node1.getEphemeralId()).thenReturn("test-node-model-match");
        DiscoveryNode node2 = mock(DiscoveryNode.class);
        when(node2.getEphemeralId()).thenReturn("test-node-not-model-match");
        List<DiscoveryNode> nodes = new ArrayList<DiscoveryNode>();
        nodes.add(node1);
        nodes.add(node2);
        doAnswer(invocationOnMock -> {
            SearchResponse searchResponse = mock(SearchResponse.class);
            SearchHits searchHits = mock(SearchHits.class);
            when(searchResponse.getHits()).thenReturn(searchHits);
            SearchHit searchHit = mock(SearchHit.class);
            when(searchHit.getId()).thenReturn(modelId);
            SearchHit[] searchHitArray = new SearchHit[1];
            searchHitArray[0] = searchHit;
            when(searchHits.getHits()).thenReturn(searchHitArray);
            ((ActionListener<SearchResponse>) invocationOnMock.getArguments()[1]).onResponse(searchResponse);
            return null;
        }).when(modelDao).search(any(SearchRequest.class), any(ActionListener.class));
        doAnswer(invocationOnMock -> { return null; }).when(modelDao).update(any(Model.class), any(ActionListener.class));

        TrainingJobClusterStateListener.initialize(threadPool, modelDao, clusterService);

        trainingJobClusterStateListener.updateModelsNodesRemoved(nodes);

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        verify(modelMetadata, times(1)).setState(ModelState.FAILED);
        verify(modelMetadata, times(1)).setError("Training failed to complete as node dropped");
        verify(modelDao, times(1)).update(any(Model.class), any(ActionListener.class));
    }
}
