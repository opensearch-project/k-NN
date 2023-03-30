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

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

/**
 * TrainingJobRunner is a singleton class responsible for submitting TrainingJobs to the k-NN training pool executor.
 * Capacity of queue and number of threads of the executor can be configured from executor construction (in KNNPlugin).
 */
@Log4j2
public class TrainingJobRunner {
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
            serializeModel(trainingJob, ActionListener.wrap(indexResponse -> {
                // Respond to the request with the initial index response
                listener.onResponse(indexResponse);
                train(trainingJob);
            }, e -> {
                // Serialization failed. Let listener handle the exception, but free up resources.
                jobCount.decrementAndGet();
                semaphore.release();
                log.error("Unable to initialize model serialization", e);
                listener.onFailure(e);
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
            indexResponse -> log.info("[KNN] Model serialization update for model [{}] was successful", trainingJob.getModelId()),
            e -> {
                log.error("[KNN] Model serialization update for model [{}] failed", trainingJob.getModelId(), e);
                KNNCounter.TRAINING_ERRORS.increment();
            }
        );

        try {
            log.info("Submitting job to training thread pool");
            threadPool.executor(TRAIN_THREAD_POOL).execute(() -> {
                try {
                    log.info("Running training job for model [{}]", trainingJob.getModelId());
                    trainingJob.run();
                    log.info("Training job for model [{}] has completed", trainingJob.getModelId());
                    serializeModel(trainingJob, loggingListener, true);
                } catch (IOException e) {
                    log.error("Unable to serialize model [{}]", trainingJob.getModelId(), e);
                    KNNCounter.TRAINING_ERRORS.increment();
                } catch (Exception e) {
                    log.error("Unable to complete training for [{}]", trainingJob.getModelId(), e);
                    KNNCounter.TRAINING_ERRORS.increment();
                } finally {
                    jobCount.decrementAndGet();
                    semaphore.release();
                }
            });
        } catch (RejectedExecutionException ree) {
            log.error("Unable to train model [{}]", trainingJob.getModelId(), ree);

            ModelMetadata modelMetadata = trainingJob.getModel().getModelMetadata();
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError("Training job execution was rejected. Node's training queue is at capacity.");

            try {
                serializeModel(trainingJob, loggingListener, true);
            } catch (IOException ioe) {
                log.error("Unable to serialize the failure for model  [{}]", trainingJob.getModelId(), ioe);
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
}
