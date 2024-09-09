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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

/**
 * TrainingJobRunner is a singleton class responsible for submitting TrainingJobs to the k-NN training pool executor.
 * Capacity of queue and number of threads of the executor can be configured from executor construction (in KNNPlugin).
 */
public class TrainingJobRunner {

    public static Logger logger = LogManager.getLogger(TrainingJobRunner.class);

    private static TrainingJobRunner INSTANCE;
    private static ModelDao modelDao;
    private static ThreadPool threadPool;

    private final Semaphore semaphore;
    private final AtomicInteger jobCount;

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
    public static void initialize(ThreadPool threadPool, ModelDao modelDao) {
        TrainingJobRunner.threadPool = threadPool;
        TrainingJobRunner.modelDao = modelDao;
    }

    /**
     * Execute a training job. This function will first grab a permit, and then serialize the initial model, then
     * execute training, and then serialize the final result.
     *
     * @param trainingJob training job to be executed
     * @param listener listener to handle final model serialization response (or exception)
     */
    public void execute(TrainingJob trainingJob, ActionListener<IndexResponse> listener) throws IOException, ExecutionException,
        InterruptedException {
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
        } catch (IOException | ExecutionException | InterruptedException e) {
            jobCount.decrementAndGet();
            semaphore.release();
            throw e;
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
                    trainingJob.run();
                    serializeModel(trainingJob, loggingListener, true);
                } catch (IOException | ExecutionException | InterruptedException e) {
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
            } catch (IOException | ExecutionException | InterruptedException e) {
                logger.error("Unable to serialize the failure for model \"{}\": ", trainingJob.getModelId(), e);
            } finally {
                jobCount.decrementAndGet();
                semaphore.release();
                KNNCounter.TRAINING_ERRORS.increment();
            }
        }
    }

    private void serializeModel(TrainingJob trainingJob, ActionListener<IndexResponse> listener, boolean update) throws IOException,
        ExecutionException, InterruptedException {
        if (update) {
            ModelMetadata modelMetadata = modelDao.getMetadata(trainingJob.getModelId());
            if (modelMetadata.getState().equals(ModelState.TRAINING)) {
                modelDao.update(trainingJob.getModel(), listener);
            } else {
                logger.info("Model state is {}. Skipping serialization of trained data", modelMetadata.getState());
            }
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
}
