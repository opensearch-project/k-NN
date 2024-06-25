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

import org.apache.commons.lang.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.UUIDs;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.index.util.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Encapsulates all information required to generate and train a model.
 */
public class TrainingJob implements Runnable {

    public static Logger logger = LogManager.getLogger(TrainingJob.class);

    private final KNNMethodContext knnMethodContext;
    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private final NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext;
    private final NativeMemoryEntryContext.AnonymousEntryContext modelAnonymousEntryContext;
    private final Model model;

    private String modelId;

    /**
     * Constructor.
     *
     * @param modelId String to identify model. If null, one will be generated.
     * @param knnMethodContext Method definition used to construct model.
     * @param nativeMemoryCacheManager Cache manager loads training data into native memory.
     * @param trainingDataEntryContext Training data configuration
     * @param modelAnonymousEntryContext Model allocation context
     * @param dimension model's dimension
     * @param description user provided description of the model.
     */
    public TrainingJob(
        String modelId,
        KNNMethodContext knnMethodContext,
        NativeMemoryCacheManager nativeMemoryCacheManager,
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext,
        NativeMemoryEntryContext.AnonymousEntryContext modelAnonymousEntryContext,
        int dimension,
        String description,
        String nodeAssignment,
        VectorDataType vectorDataType
    ) {
        // Generate random base64 string if one is not provided
        this.modelId = StringUtils.isNotBlank(modelId) ? modelId : UUIDs.randomBase64UUID();
        this.knnMethodContext = Objects.requireNonNull(knnMethodContext, "MethodContext cannot be null.");
        this.nativeMemoryCacheManager = Objects.requireNonNull(nativeMemoryCacheManager, "NativeMemoryCacheManager cannot be null.");
        this.trainingDataEntryContext = Objects.requireNonNull(trainingDataEntryContext, "TrainingDataEntryContext cannot be null.");
        this.modelAnonymousEntryContext = Objects.requireNonNull(modelAnonymousEntryContext, "AnonymousEntryContext cannot be null.");
        this.model = new Model(
            new ModelMetadata(
                knnMethodContext.getKnnEngine(),
                knnMethodContext.getSpaceType(),
                dimension,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                description,
                "",
                nodeAssignment,
                knnMethodContext.getMethodComponentContext(),
                vectorDataType
            ),
            null,
            this.modelId
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
     * Getter for model
     *
     * @return model
     */
    public Model getModel() {
        return model;
    }

    @Override
    public void run() {
        NativeMemoryAllocation trainingDataAllocation = null;
        NativeMemoryAllocation modelAnonymousAllocation = null;
        ModelMetadata modelMetadata = model.getModelMetadata();

        try {
            // Get training data
            trainingDataAllocation = nativeMemoryCacheManager.get(trainingDataEntryContext, false);

            // Acquire lock on allocation -- this will wait until training data is loaded
            trainingDataAllocation.readLock();
        } catch (Exception e) {
            logger.error("Failed to get training data for model \"" + modelId + "\": " + e.getMessage());
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError(
                "Failed to load training data into memory. " + "Check if there is enough memory to perform the request."
            );

            if (trainingDataAllocation != null) {
                nativeMemoryCacheManager.invalidate(trainingDataEntryContext.getKey());
            }

            KNNCounter.TRAINING_ERRORS.increment();

            return;
        }

        try {
            // Reserve space in the cache for the model
            modelAnonymousAllocation = nativeMemoryCacheManager.get(modelAnonymousEntryContext, false);

            // Lock until training completes
            modelAnonymousAllocation.readLock();
        } catch (Exception e) {
            logger.error("Failed to allocate space in native memory for model \"" + modelId + "\": " + e.getMessage());
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError(
                "Failed to allocate space in native memory for the model. " + "Check if there is enough memory to perform the request."
            );

            trainingDataAllocation.readUnlock();
            nativeMemoryCacheManager.invalidate(trainingDataEntryContext.getKey());

            if (modelAnonymousAllocation != null) {
                nativeMemoryCacheManager.invalidate(modelAnonymousEntryContext.getKey());
            }

            KNNCounter.TRAINING_ERRORS.increment();

            return;
        }

        // Once locks are acquired, train the model. We need a separate try/catch block due to the fact that the lock
        // needs to be released after they are acquired, but cannot be released if it has not been acquired.
        try {
            // We need to check if either allocation is closed before we proceed. There is a possibility that
            // immediately after the cache returns the allocation, it will grab the write lock and close them before
            // this method can get the read lock.
            if (modelAnonymousAllocation.isClosed()) {
                throw new RuntimeException("Unable to reserve memory for model: allocation is already closed");
            }

            if (trainingDataAllocation.isClosed()) {
                throw new RuntimeException("Unable to load training data into memory: allocation is already closed");
            }
            setVersionInKnnMethodContext();
            Map<String, Object> trainParameters = model.getModelMetadata().getKnnEngine().getMethodAsMap(knnMethodContext);
            trainParameters.put(
                KNNConstants.INDEX_THREAD_QTY,
                KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY)
            );

            if (VectorDataType.BINARY == model.getModelMetadata().getVectorDataType()) {
                trainParameters.put(
                    KNNConstants.INDEX_DESCRIPTION_PARAMETER,
                    FAISS_BINARY_INDEX_DESCRIPTION_PREFIX + trainParameters.get(KNNConstants.INDEX_DESCRIPTION_PARAMETER).toString()
                );
            }

            IndexUtil.updateVectorDataTypeToParameters(trainParameters, model.getModelMetadata().getVectorDataType());

            byte[] modelBlob = JNIService.trainIndex(
                trainParameters,
                model.getModelMetadata().getDimension(),
                trainingDataAllocation.getMemoryAddress(),
                model.getModelMetadata().getKnnEngine()
            );

            // Once training finishes, update model
            model.setModelBlob(modelBlob);
            modelMetadata.setState(ModelState.CREATED);
        } catch (Exception e) {
            logger.error("Failed to run training job for model \"" + modelId + "\": ", e);
            modelMetadata.setState(ModelState.FAILED);
            modelMetadata.setError(
                "Failed to execute training. May be caused by an invalid method definition or " + "not enough memory to perform training."
            );

            KNNCounter.TRAINING_ERRORS.increment();

        } finally {
            // Invalidate right away so we dont run into any big memory problems
            trainingDataAllocation.readUnlock();
            modelAnonymousAllocation.readUnlock();
            nativeMemoryCacheManager.invalidate(trainingDataEntryContext.getKey());
            nativeMemoryCacheManager.invalidate(modelAnonymousEntryContext.getKey());
        }
    }

    private void setVersionInKnnMethodContext() {
        // We are picking up the node version here. For more details why we did this please check below conversation
        // Ref: https://github.com/opensearch-project/k-NN/pull/1353#discussion_r1434428542
        knnMethodContext.getMethodComponentContext().setIndexVersion(trainingDataEntryContext.getClusterService().localNode().getVersion());
    }
}
