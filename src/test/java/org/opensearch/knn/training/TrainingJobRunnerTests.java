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
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.core.index.shard.ShardId;
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
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

public class TrainingJobRunnerTests extends KNNTestCase {

    @SuppressWarnings("unchecked")
    public void testExecute_success() throws IOException, InterruptedException {
        // Test makes sure the correct execution logic follows on successful run
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        when(model.getModelMetadata()).thenReturn(mock(ModelMetadata.class));
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);
        doAnswer(invocationOnMock -> null).when(trainingJob).run();

        // This gets called right after the initial put, before training begins. Just check that the model id is
        // equal
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(
            indexResponse -> assertEquals(modelId, indexResponse.getId()),
            e -> fail("Failure should not have occurred")
        );

        // After put finishes, it should call the onResponse function that will call responseListener and then kickoff
        // training.
        ModelDao modelDao = mock(ModelDao.class);
        doAnswer(invocationOnMock -> {
            assertEquals(1, trainingJobRunner.getJobCount()); // Make sure job count is correct
            IndexResponse indexResponse = new IndexResponse(new ShardId(MODEL_INDEX_NAME, "uuid", 0), modelId, 0, 0, 0, true);
            ((ActionListener<IndexResponse>) invocationOnMock.getArguments()[1]).onResponse(indexResponse);
            return null;
        }).when(modelDao).put(any(Model.class), any(ActionListener.class));

        // Function finishes when update is called
        doAnswer(invocationOnMock -> null).when(modelDao).update(any(Model.class), any(ActionListener.class));

        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.localNode()).thenReturn(mock(DiscoveryNode.class));

        // Finally, initialize the singleton runner, execute the job.
        TrainingJobRunner.initialize(threadPool, modelDao, clusterService);
        trainingJobRunner.execute(trainingJob, responseListener);

        // Immediately, we shutdown the executor and await its termination.
        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        // Make sure these methods get called once
        verify(trainingJob, times(1)).run();
        verify(modelDao, times(1)).put(any(Model.class), any(ActionListener.class));
        verify(modelDao, times(1)).update(any(Model.class), any(ActionListener.class));
    }

    @SuppressWarnings("unchecked")
    public void testExecute_failure_rejected() throws IOException, InterruptedException {
        // This test makes sure we reject another request when one is ongoing. To do this, we call
        // trainingJobRunner.execute(trainingJob, responseListener) in the mocked modeldao.update. At this point,
        // the call should produce a failure because a training job is already ongoing.

        ExecutorService executorService = Executors.newSingleThreadExecutor();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        when(model.getModelMetadata()).thenReturn(mock(ModelMetadata.class));
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);
        doAnswer(invocationOnMock -> null).when(trainingJob).run();

        // This gets called right after the initial put, before training begins. Just check that the model id is
        // equal
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(
            indexResponse -> assertEquals(modelId, indexResponse.getId()),
            e -> fail("Should not reach this state")
        );

        // After put finishes, it should call the onResponse function that will call responseListener and then kickoff
        // training.
        ModelDao modelDao = mock(ModelDao.class);
        doAnswer(invocationOnMock -> {
            IndexResponse indexResponse = new IndexResponse(new ShardId(MODEL_INDEX_NAME, "uuid", 0), modelId, 0, 0, 0, true);
            ((ActionListener<IndexResponse>) invocationOnMock.getArguments()[1]).onResponse(indexResponse);
            return null;
        }).when(modelDao).put(any(Model.class), any(ActionListener.class));

        // Once update is called, try to start another training job. This should fail because the calling thread
        // is running training
        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();
        doAnswer(
            invocationOnMock -> expectThrows(
                RejectedExecutionException.class,
                () -> trainingJobRunner.execute(trainingJob, responseListener)
            )
        ).when(modelDao).update(model, responseListener);

        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.localNode()).thenReturn(mock(DiscoveryNode.class));

        // Finally, initialize the singleton runner, execute the job.
        TrainingJobRunner.initialize(threadPool, modelDao, clusterService);
        trainingJobRunner.execute(trainingJob, responseListener);

        // Immediately, we shutdown the executor and await its termination.
        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    }

    public void testClusterChanged() throws InterruptedException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);
        doAnswer(invocationOnMock -> { return null; }).when(threadPool)
            .schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        ModelDao modelDao = mock(ModelDao.class);
        ClusterChangedEvent clusterChangedEvent = mock(ClusterChangedEvent.class);
        when(clusterChangedEvent.localNodeClusterManager()).thenReturn(true);
        when(clusterChangedEvent.isNewCluster()).thenReturn(true);

        TrainingJobRunner.initialize(threadPool, modelDao, clusterService);

        trainingJobRunner.clusterChanged(clusterChangedEvent);

        verify(threadPool, times(1)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        when(clusterChangedEvent.isNewCluster()).thenReturn(false);
        when(clusterChangedEvent.nodesRemoved()).thenReturn(true);
        DiscoveryNodes.Delta delta = mock(DiscoveryNodes.Delta.class);
        List<DiscoveryNode> nodes = new ArrayList<>();
        when(clusterChangedEvent.nodesDelta()).thenReturn(delta);
        when(delta.removedNodes()).thenReturn(nodes);

        trainingJobRunner.clusterChanged(clusterChangedEvent);

        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));
        verify(clusterChangedEvent, times(1)).nodesDelta();

        when(clusterChangedEvent.nodesRemoved()).thenReturn(false);
        trainingJobRunner.clusterChanged(clusterChangedEvent);
        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        when(clusterChangedEvent.localNodeClusterManager()).thenReturn(false);
        trainingJobRunner.clusterChanged(clusterChangedEvent);
        verify(threadPool, times(2)).schedule(any(Runnable.class), any(TimeValue.class), any(String.class));

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);
    }

    public void testUpdateModelsNewCluster() throws IOException, InterruptedException, ExecutionException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

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

        TrainingJobRunner.initialize(threadPool, modelDao, clusterService);

        trainingJobRunner.updateModelsNewCluster();

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        verify(modelMetadata, times(1)).setState(ModelState.FAILED);
        verify(modelMetadata, times(1)).setError("Training failed to complete due to node drop");
        verify(modelDao, times(1)).update(any(Model.class), any(ActionListener.class));
    }

    public void testUpdateModelsNodesRemoved() throws IOException, InterruptedException, ExecutionException {
        ExecutorService executorService = Executors.newSingleThreadExecutor();

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

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

        TrainingJobRunner.initialize(threadPool, modelDao, clusterService);

        trainingJobRunner.updateModelsNodesRemoved(nodes);

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        verify(modelMetadata, times(1)).setState(ModelState.ZOMBIE);
        verify(modelMetadata, times(1)).setError("A node dropped and left the model training process in a zombie state");
        verify(modelDao, times(1)).update(any(Model.class), any(ActionListener.class));
    }
}
