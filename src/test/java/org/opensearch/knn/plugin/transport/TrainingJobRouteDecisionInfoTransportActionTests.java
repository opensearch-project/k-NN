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

package org.opensearch.knn.plugin.transport;

import org.junit.After;
import org.junit.Before;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.training.TrainingJob;
import org.opensearch.knn.training.TrainingJobRunner;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

public class TrainingJobRouteDecisionInfoTransportActionTests extends KNNSingleNodeTestCase {

    ExecutorService executorService;

    @Before
    public void setup() {
        executorService = Executors.newSingleThreadExecutor();
    }

    @After
    public void teardown() {
        executorService.shutdown();
    }

    @SuppressWarnings("unchecked")
    public void testNodeOperation() throws IOException, InterruptedException {
        // Ensure initial value of train job count is 0
        TrainingJobRouteDecisionInfoTransportAction action = node().injector()
            .getInstance(TrainingJobRouteDecisionInfoTransportAction.class);

        TrainingJobRouteDecisionInfoNodeRequest request = new TrainingJobRouteDecisionInfoNodeRequest();

        TrainingJobRouteDecisionInfoNodeResponse response1 = action.nodeOperation(request);
        assertEquals(0, response1.getTrainingJobCount().intValue());

        // Setup mocked training job
        String modelId = "model-id";
        Model model = mock(Model.class);
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);
        doAnswer(invocationOnMock -> null).when(trainingJob).run();

        ModelDao modelDao = mock(ModelDao.class);

        // Here we check to make sure there is a running job
        doAnswer(invocationOnMock -> {
            TrainingJobRouteDecisionInfoNodeResponse response2 = action.nodeOperation(request);
            assertEquals(1, response2.getTrainingJobCount().intValue());

            IndexResponse indexResponse = new IndexResponse(new ShardId(MODEL_INDEX_NAME, "uuid", 0), modelId, 0, 0, 0, true);
            ((ActionListener<IndexResponse>) invocationOnMock.getArguments()[1]).onResponse(indexResponse);
            return null;
        }).when(modelDao).put(any(Model.class), any(ActionListener.class));

        // Set up the rest of the training logic
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(
            indexResponse -> { inProgressLatch.countDown(); },
            e -> fail("Failure should not have occurred")
        );

        doAnswer(invocationOnMock -> {
            responseListener.onResponse(mock(IndexResponse.class));
            return null;
        }).when(modelDao).update(model, responseListener);

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        // Initialize runner and execute job
        TrainingJobRunner.initialize(threadPool, modelDao);
        TrainingJobRunner.getInstance().execute(trainingJob, responseListener);

        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }
}
