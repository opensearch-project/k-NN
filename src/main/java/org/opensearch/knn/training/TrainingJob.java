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
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.util.Objects;

/**
 * Encapsulates all information required to generate and train a model.
 */
public class TrainingJob implements Runnable {

    public static Logger logger = LogManager.getLogger(TrainingJob.class);

    private final KNNMethodContext knnMethodContext;
    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private final NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext;
    private final Model model;

    private String modelId;

    /**
     * Constructor.
     *
     * @param modelId String to identify model. If null, one will be generated.
     * @param knnMethodContext Method definition used to construct model.
     * @param nativeMemoryCacheManager Cache manager loads training data into native memory.
     * @param trainingDataEntryContext Training data configuration
     * @param dimension model's dimension
     * @param description user provided description of the model.
     */
    public TrainingJob(String modelId, KNNMethodContext knnMethodContext,
                       NativeMemoryCacheManager nativeMemoryCacheManager,
                       NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext,
                       int dimension, String description) {
        this.modelId = modelId;
        this.knnMethodContext = Objects.requireNonNull(knnMethodContext, "MethodContext cannot be null.");
        this.nativeMemoryCacheManager = Objects.requireNonNull(nativeMemoryCacheManager,
                "NativeMemoryCacheManager cannot be null.");
        this.trainingDataEntryContext = Objects.requireNonNull(trainingDataEntryContext,
                "TrainingDataEntryContext cannot be null.");
        this.model = new Model(
                        new ModelMetadata(
                                knnMethodContext.getEngine(),
                                knnMethodContext.getSpaceType(),
                                dimension,
                                ModelState.TRAINING,
                                TimeValue.timeValueMillis(System.currentTimeMillis()),
                                description,
                                ""
                        ),
                        null
                    );
    }

    /**
     * Getter for model id.
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Setter for model id.
     *
     * @param modelId to set
     */
    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    /**
     * Getter for model
     *
     * @return model
     */
    public Model getModel() {
        return model;
    }

    @Override
    public void run() {
        NativeMemoryAllocation nativeMemoryAllocation = null;
        ModelMetadata modelMetadata = model.getModelMetadata();

        try {
            // Get training data
            nativeMemoryAllocation = nativeMemoryCacheManager.get(trainingDataEntryContext, false);

            // Acquire lock on allocation -- this will wait until training data is loaded
            nativeMemoryAllocation.readLock();
        } catch (Exception e) {
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError(e.getMessage());

            if (nativeMemoryAllocation != null) {
                nativeMemoryCacheManager.invalidate(trainingDataEntryContext.getKey());
            }

            logger.error("Failed to get training data for model \"" + modelId + "\": " + modelMetadata.getError());
            return;
        }

        // Once lock is acquired, train the model. We need a separate try/catch block due to the fact that the lock
        // needs to be released after it is acquired, but cannot be released if it has not been acquired.
        try {
            if (nativeMemoryAllocation.isClosed()) {
                throw new RuntimeException("Unable to load training data into memory: allocation is already closed");
            }

            byte[] modelBlob = JNIService.trainIndex(
                    model.getModelMetadata().getKnnEngine().getMethodAsMap(knnMethodContext),
                    model.getModelMetadata().getDimension(),
                    nativeMemoryAllocation.getMemoryAddress(),
                    model.getModelMetadata().getKnnEngine().getName()
            );

            // Once training finishes, update model
            model.setModelBlob(modelBlob);
            modelMetadata.setState(ModelState.CREATED);
        } catch (Exception e) {
            modelMetadata.setState(ModelState.FAILED);
            logger.error("Exception \"" + modelId + "\": " + e);
            modelMetadata.setError(e.getMessage());
            nativeMemoryAllocation.readUnlock();
            logger.error("Failed to run training job for model \"" + modelId + "\": " + modelMetadata.getError());
        } finally {
            // Invalidate right away so we dont run into any big memory problems
            nativeMemoryCacheManager.invalidate(trainingDataEntryContext.getKey());
            nativeMemoryAllocation.readUnlock();
        }
    }
}
