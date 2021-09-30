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

import org.junit.After;
import org.junit.Before;
import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.index.shard.ShardId;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

public class TrainingJobRunnerTests extends KNNTestCase {

    ExecutorService executorService = Executors.newSingleThreadExecutor();

    @Before
    public void setup() {
        executorService = Executors.newSingleThreadExecutor();
    }

    @After
    public void teardown() {
        executorService.shutdown();
    }

    @SuppressWarnings("unchecked")
    public void testExecute_success() throws IOException, InterruptedException {
        // Test makes sure the correct execution logic follows on successful run

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);
        doAnswer(invocationOnMock -> null).when(trainingJob).setModelId(modelId);
        doAnswer(invocationOnMock -> null).when(trainingJob).run();

        // Return a result for put with modelId as well as created set to true. For update, same
        // thing except created should be false
        ModelDao modelDao = mock(ModelDao.class);
        doAnswer(invocationOnMock -> {
            assertEquals(1, trainingJobRunner.getJobCount()); // Make sure job count is correct
            IndexResponse indexResponse = new IndexResponse(
                    new ShardId(MODEL_INDEX_NAME, "uuid", 0),
                    "any-type",
                    modelId,
                    0,
                    0,
                    0,
                    true
                    );
            ((ActionListener<IndexResponse>)invocationOnMock.getArguments()[2]).onResponse(indexResponse);
            return null;
        }).when(modelDao).put(anyString(), any(Model.class), any(ActionListener.class));

        // All validation will need to be done in this listener
        // On successful allocation, the response should return success. This listener will be called after the update
        // finishes
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(indexResponse -> {
            assertEquals(modelId, indexResponse.getId());
            inProgressLatch.countDown();
        }, e -> {
            fail("Failure should not have occurred");
        });

        doAnswer(invocationOnMock -> {
            IndexResponse indexResponse = new IndexResponse(
                    new ShardId(MODEL_INDEX_NAME, "uuid", 0),
                    "any-type",
                    modelId,
                    0,
                    0,
                    0,
                    false
            );
            responseListener.onResponse(indexResponse);
            return null;
        }).when(modelDao).update(modelId, model, responseListener);

        // Finally, initialize the singleton runner, execute the job.
        TrainingJobRunner.initialize(threadPool, modelDao);

        trainingJobRunner.execute(trainingJob, responseListener);
        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));

        // Make sure these methods get called once
        verify(trainingJob, times(1)).setModelId(modelId);
        verify(trainingJob, times(1)).run();
        verify(modelDao, times(1))
                .put(anyString(), any(Model.class), any(ActionListener.class));
        verify(modelDao, times(1)).update(modelId, model, responseListener);
    }

    @SuppressWarnings("unchecked")
    public void testExecute_failure_rejected() throws IOException, InterruptedException {
        // This test makes sure we reject another request when one is ongoing. To do this, we call
        // trainingJobRunner.execute(trainingJob, responseListener) in the mocked modeldao.update. At this point,
        // the call should produce a failure because a training job is already ongoing.

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);
        doAnswer(invocationOnMock -> null).when(trainingJob).setModelId(modelId);
        doAnswer(invocationOnMock -> null).when(trainingJob).run();

        // Return a result for modelDao put with modelId as well as created set to true. For update, same
        // thing except created should be false
        ModelDao modelDao = mock(ModelDao.class);
        doAnswer(invocationOnMock -> {
            IndexResponse indexResponse = new IndexResponse(
                    new ShardId(MODEL_INDEX_NAME, "uuid", 0),
                    "any-type",
                    modelId,
                    0,
                    0,
                    0,
                    true
            );
            ((ActionListener<IndexResponse>)invocationOnMock.getArguments()[2]).onResponse(indexResponse);
            return null;
        }).when(modelDao).put(anyString(), any(Model.class), any(ActionListener.class));

        // No-op listener
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(indexResponse -> {
            inProgressLatch.countDown();
        }, e -> {
            fail("Should not reach this state");
        });

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();
        doAnswer(invocationOnMock -> {
            expectThrows(RejectedExecutionException.class, () -> trainingJobRunner.execute(trainingJob, responseListener));
            responseListener.onResponse(null);
            return null;
        }).when(modelDao).update(modelId, model, responseListener);

        // Finally, initialize the singleton runner, execute the job.
        TrainingJobRunner.initialize(threadPool, modelDao);
        trainingJobRunner.execute(trainingJob, responseListener);

        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }

    @SuppressWarnings("unchecked")
    public void testExecute_failure_serialization() throws IOException, InterruptedException {
        // This test confirms that execution fails as expected if initial serialization fails

        TrainingJobRunner trainingJobRunner = TrainingJobRunner.getInstance();

        ThreadPool threadPool = mock(ThreadPool.class);
        when(threadPool.executor(TRAIN_THREAD_POOL)).thenReturn(executorService);

        String modelId = "test-model-id";
        Model model = mock(Model.class);
        TrainingJob trainingJob = mock(TrainingJob.class);
        when(trainingJob.getModelId()).thenReturn(modelId);
        when(trainingJob.getModel()).thenReturn(model);

        // Listener should validate exception comes through
        String message = "some error";
        final CountDownLatch inProgressLatch = new CountDownLatch(1);
        ActionListener<IndexResponse> responseListener = ActionListener.wrap(
                indexResponse -> fail("Should not reach this state"),
                e -> {
                    assertEquals(e.getMessage(), message);
                    assertEquals(0, trainingJobRunner.getJobCount()); // Make sure resources are free
                    inProgressLatch.countDown();
                });

        // ModelDao put should just call listeners onFailure
        ModelDao modelDao = mock(ModelDao.class);
        doAnswer(invocationOnMock -> {
            ((ActionListener<IndexResponse>)invocationOnMock.getArguments()[2]).onFailure(new RuntimeException(message));
            return null;
        }).when(modelDao).put(anyString(), any(Model.class), any(ActionListener.class));



        // Finally, initialize the singleton runner, execute the job.
        TrainingJobRunner.initialize(threadPool, modelDao);
        trainingJobRunner.execute(trainingJob, responseListener);

        assertTrue(inProgressLatch.await(100, TimeUnit.SECONDS));
    }
}
