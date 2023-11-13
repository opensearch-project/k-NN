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

import lombok.SneakyThrows;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.search.SearchHit;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

/**
 * TrainingJobRunner is a singleton class responsible for submitting TrainingJobs to the k-NN training pool executor.
 * Capacity of queue and number of threads of the executor can be configured from executor construction (in KNNPlugin).
 */
public class TrainingJobRunner implements ClusterStateListener {

    public static Logger logger = LogManager.getLogger(TrainingJobRunner.class);

    private static TrainingJobRunner INSTANCE;
    private static ModelDao modelDao;
    private static ThreadPool threadPool;

    private final Semaphore semaphore;
    private final AtomicInteger jobCount;
    private static ClusterService clusterService;

    /**
     * Get singleton instance of TrainingJobRunner
     *
     * @return singleton instance of TrainingJobRunner
     */
    public static synchronized TrainingJobRunner getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new TrainingJobRunner();
        }
        return INSTANCE;
    }

    private TrainingJobRunner() {
        this.jobCount = new AtomicInteger(0);
        this.semaphore = new Semaphore(1);
    }

    /**
     * Initializes static components.
     *
     * @param threadPool threadPool to use to get KNN Training Executor
     * @param modelDao modelDao used to serialize the models
     */
    public static void initialize(ThreadPool threadPool, ModelDao modelDao, ClusterService clusterService) {
        TrainingJobRunner.threadPool = threadPool;
        TrainingJobRunner.modelDao = modelDao;
        TrainingJobRunner.clusterService = clusterService;
        clusterService.addListener(TrainingJobRunner.getInstance());
    }

    /**
     * Execute a training job. This function will first grab a permit, and then serialize the initial model, then
     * execute training, and then serialize the final result.
     *
     * @param trainingJob training job to be executed
     * @param listener listener to handle final model serialization response (or exception)
     */
    public void execute(TrainingJob trainingJob, ActionListener<IndexResponse> listener) throws IOException {
        // If the semaphore cannot be acquired, the node is unable to execute this job. This allows us to limit
        // the number of training jobs that enter this function. Although the training threadpool size will also prevent
        // this, we want to prevent this before we perform any serialization.
        if (!semaphore.tryAcquire()) {
            ValidationException exception = new ValidationException();
            exception.addValidationError("Unable to run training job: No training capacity on node.");
            KNNCounter.TRAINING_ERRORS.increment();
            throw exception;
        }

        jobCount.incrementAndGet();

        // Serialize model before training. The model should be in the training state and the model binary should be
        // null. This notifies users that their model is training, but not yet ready for use.
        try {
            trainingJob.getModel().getModelMetadata().setNodeAssignment(clusterService.localNode().getEphemeralId());
            serializeModel(trainingJob, ActionListener.wrap(indexResponse -> {
                // Respond to the request with the initial index response
                listener.onResponse(indexResponse);
                train(trainingJob);
            }, exception -> {
                // Serialization failed. Let listener handle the exception, but free up resources.
                jobCount.decrementAndGet();
                semaphore.release();
                logger.error("Unable to initialize model serialization: " + exception.getMessage());
                listener.onFailure(exception);
            }), false);
        } catch (IOException ioe) {
            jobCount.decrementAndGet();
            semaphore.release();
            throw ioe;
        }
    }

    private void train(TrainingJob trainingJob) {
        // Attempt to submit job to training thread pool. On failure, release the resources and serialize the failure.

        // Listener for update model after training index action
        ActionListener<IndexResponse> loggingListener = ActionListener.wrap(
            indexResponse -> logger.debug("[KNN] Model serialization update for \"" + trainingJob.getModelId() + "\" was successful"),
            e -> {
                logger.error("[KNN] Model serialization update for \"" + trainingJob.getModelId() + "\" failed: " + e.getMessage());
                KNNCounter.TRAINING_ERRORS.increment();
            }
        );

        try {
            threadPool.executor(TRAIN_THREAD_POOL).execute(() -> {
                try {
                    Thread.sleep(300*1000);
                    trainingJob.run();
                    serializeModel(trainingJob, loggingListener, true);
                } catch (IOException e) {
                    logger.error("Unable to serialize model \"" + trainingJob.getModelId() + "\": " + e.getMessage());
                    KNNCounter.TRAINING_ERRORS.increment();
                } catch (Exception e) {
                    logger.error("Unable to complete training for \"" + trainingJob.getModelId() + "\": " + e.getMessage());
                    KNNCounter.TRAINING_ERRORS.increment();
                } finally {
                    jobCount.decrementAndGet();
                    semaphore.release();
                }
            });
        } catch (RejectedExecutionException ree) {
            logger.error("Unable to train model \"" + trainingJob.getModelId() + "\": " + ree.getMessage());

            ModelMetadata modelMetadata = trainingJob.getModel().getModelMetadata();
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError("Training job execution was rejected. Node's training queue is at capacity.");

            try {
                serializeModel(trainingJob, loggingListener, true);
            } catch (IOException ioe) {
                logger.error("Unable to serialize the failure for model \"" + trainingJob.getModelId() + "\": " + ioe);
            } finally {
                jobCount.decrementAndGet();
                semaphore.release();
                KNNCounter.TRAINING_ERRORS.increment();
            }
        }
    }

    private void serializeModel(TrainingJob trainingJob, ActionListener<IndexResponse> listener, boolean update) throws IOException {
        if (update) {
            modelDao.update(trainingJob.getModel(), listener);
        } else {
            modelDao.put(trainingJob.getModel(), listener);
        }
    }

    /**
     * Get all jobs in the runner.
     *
     * @return number of running jobs.
     */
    public int getJobCount() {
        return jobCount.get();
    }

    @SneakyThrows
    @Override
    public void clusterChanged(ClusterChangedEvent event) {
        if (event.localNodeClusterManager()) {
            if (event.isNewCluster()) {
                threadPool.schedule(this::updateModelsNewCluster, TimeValue.timeValueSeconds(1), ThreadPool.Names.GENERIC);
            } else if (event.nodesRemoved()) {
                List<DiscoveryNode> removedNodes = event.nodesDelta().removedNodes();
                updateModelsNodesRemoved(removedNodes);
            }
        }
    }

    @SneakyThrows
    public void updateModelsNewCluster() {
        if (modelDao.isCreated()) {
            List<String> modelIds = searchModelIds();
            for (String modelId : modelIds) {
                Model model = modelDao.get(modelId);
                ModelMetadata modelMetadata = model.getModelMetadata();
                if (modelMetadata.getState().equals(ModelState.TRAINING)) {
                    modelMetadata.setState(ModelState.FAILED);
                    modelDao.update(model, new ActionListener<IndexResponse>() {
                        @Override
                        public void onResponse(IndexResponse indexResponse) {
                            System.out.println("Model updated successfully");
                        }

                        @Override
                        public void onFailure(Exception e) {
                            System.out.println("Model not updated");
                        }
                    });
                }
            }
        }
    }

    @SneakyThrows
    public void updateModelsNodesRemoved(List<DiscoveryNode> removedNodes) {
        if (modelDao.isCreated()) {
            List<String> modelIds = searchModelIds();
            for (DiscoveryNode removedNode : removedNodes) {
                for (String modelId : modelIds) {
                    Model model = modelDao.get(modelId);
                    ModelMetadata modelMetadata = model.getModelMetadata();
                    if (modelMetadata.getNodeAssignment().equals(removedNode.getEphemeralId()) && modelMetadata.getState().equals(ModelState.TRAINING)) {
                        modelMetadata.setState(ModelState.FAILED);
                        modelDao.update(model, new ActionListener<IndexResponse>() {
                            @Override
                            public void onResponse(IndexResponse indexResponse) {
                                System.out.println("Model updated successfully");
                            }

                            @Override
                            public void onFailure(Exception e) {
                                System.out.println("Model not updated");
                            }
                        });
                    }
                }
            }
        }
    }

    @SneakyThrows
    private List<String> searchModelIds() {
        List<String> modelIds = new ArrayList<String>();
        CountDownLatch latch = new CountDownLatch(1);
        modelDao.search(new SearchRequest(), new ActionListener<SearchResponse>() {
            @Override
            public void onResponse(SearchResponse searchResponse) {
                for (SearchHit searchHit : searchResponse.getHits().getHits()) {
                    modelIds.add(searchHit.getId());
                }
                latch.countDown();
            }

            @Override
            public void onFailure(Exception e) {
                latch.countDown();
            }
        });
        latch.await();
        return modelIds;
    }
}
