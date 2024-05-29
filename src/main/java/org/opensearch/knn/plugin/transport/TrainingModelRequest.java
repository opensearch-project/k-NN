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

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.io.IOException;

/**
 * Request to train and serialize a model
 */
public class TrainingModelRequest extends ActionRequest {

    private static ClusterService clusterService;
    private static ModelDao modelDao;

    private final String modelId;
    private final KNNMethodContext knnMethodContext;
    private final int dimension;
    private final String trainingIndex;
    private final String trainingField;
    private final String preferredNodeId;
    private final String description;

    private int maximumVectorCount;
    private int searchSize;

    private int trainingDataSizeInKB;

    /**
     * Constructor.
     *
     * @param modelId for model to be created via training. If  null, an ID will be generated.
     * @param knnMethodContext method definition of model to be created
     * @param dimension for model to be created
     * @param trainingIndex OpenSearch index storing the training data
     * @param trainingField OpenSearch field storing the trianing data
     * @param preferredNodeId Preferred node to execute training on. If null, the plugin will select the node.
     * @param description User provided description of their model
     */
    public TrainingModelRequest(
        String modelId,
        KNNMethodContext knnMethodContext,
        int dimension,
        String trainingIndex,
        String trainingField,
        String preferredNodeId,
        String description
    ) {
        super();
        this.modelId = modelId;
        this.knnMethodContext = knnMethodContext;
        this.dimension = dimension;
        this.trainingIndex = trainingIndex;
        this.trainingField = trainingField;
        this.preferredNodeId = preferredNodeId;
        this.description = description;

        // Set these as defaults initially. If call wants to override them, they can use the setters.
        this.maximumVectorCount = Integer.MAX_VALUE; // By default, get all vectors in the index
        this.searchSize = 10_000; // By default, use the maximum search size

        // Training data size in kilobytes. By default, this is invalid (it cant have negative kb). It eventually gets
        // calculated in transit. A user cannot set this value directly.
        this.trainingDataSizeInKB = -1;
    }

    /**
     * Constructor from stream.
     *
     * @param in StreamInput to construct request from.
     * @throws IOException thrown when reading from stream fails
     */
    public TrainingModelRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readOptionalString();
        this.knnMethodContext = new KNNMethodContext(in);
        this.trainingIndex = in.readString();
        this.trainingField = in.readString();
        this.preferredNodeId = in.readOptionalString();
        this.dimension = in.readInt();
        this.description = in.readOptionalString();
        this.maximumVectorCount = in.readInt();
        this.searchSize = in.readInt();
        this.trainingDataSizeInKB = in.readInt();
    }

    /**
     * Initialize components of the request that are needed, but should not be passed from node to node.
     *
     * @param modelDao used to get information about models during validation
     * @param clusterService used to get information about indices during validation
     */
    public static void initialize(ModelDao modelDao, ClusterService clusterService) {
        TrainingModelRequest.modelDao = modelDao;
        TrainingModelRequest.clusterService = clusterService;
    }

    /**
     * Getter for modelId
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Getter for knnMethodContext
     *
     * @return knnMethodContext
     */
    public KNNMethodContext getKnnMethodContext() {
        return knnMethodContext;
    }

    /**
     * Getter for dimension
     *
     * @return dimension
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Getter for trainingIndex
     *
     * @return trainingIndex
     */
    public String getTrainingIndex() {
        return trainingIndex;
    }

    /**
     * Getter for trainingField
     *
     * @return trainingField
     */
    public String getTrainingField() {
        return trainingField;
    }

    /**
     * Getter for preferredNodeId
     *
     * @return preferredNodeId
     */
    public String getPreferredNodeId() {
        return preferredNodeId;
    }

    /**
     * Getter description of the model
     *
     * @return description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Getter for maximum vector count. This corresponds to the maximum number of vectors from the training index
     * a user wants to use for training.
     *
     * @return maximumVectorCount
     */
    public int getMaximumVectorCount() {
        return maximumVectorCount;
    }

    /**
     * Setter for maximum vector count
     *
     * @param maximumVectorCount to be set. It must be greater than 0
     */
    public void setMaximumVectorCount(int maximumVectorCount) {
        if (maximumVectorCount <= 0) {
            throw new IllegalArgumentException(
                String.format("Maximum vector count %d is invalid. Maximum vector " + "count must be greater than 0", maximumVectorCount)
            );
        }
        this.maximumVectorCount = maximumVectorCount;
    }

    /**
     * Getter for search size. This value corresponds to how many vectors are pulled from the training index per
     * search request
     *
     * @return searchSize
     */
    public int getSearchSize() {
        return searchSize;
    }

    /**
     * Setter for search size.
     *
     * @param searchSize to be set. Must be greater than 0 and less than 10,000
     */
    public void setSearchSize(int searchSize) {
        if (searchSize <= 0 || searchSize > 10000) {
            throw new IllegalArgumentException(
                String.format("Search size %d is invalid. Search size must be " + "between 0 and 10,000", searchSize)
            );
        }
        this.searchSize = searchSize;
    }

    /**
     * Getter for training data size in kilobytes.
     *
     * @return trainingDataSizeInKB
     */
    public int getTrainingDataSizeInKB() {
        return trainingDataSizeInKB;
    }

    /**
     * Setter for trainingDataSizeInKB. Package private to prevent users from changing this value directly.
     *
     * @param trainingDataSizeInKB to be set.
     */
    void setTrainingDataSizeInKB(int trainingDataSizeInKB) {
        if (trainingDataSizeInKB <= 0) {
            throw new IllegalArgumentException(
                String.format("Training data size %d is invalid. Training data size " + "must be greater than 0", trainingDataSizeInKB)
            );
        }
        this.trainingDataSizeInKB = trainingDataSizeInKB;
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        // Check if model id exists via model metadata
        // Also, check if model is not in model graveyard to make sure it is not being deleted
        if (modelDao.getMetadata(modelId) != null && !modelDao.isModelInGraveyard(modelId)) {
            exception = new ActionRequestValidationException();
            exception.addValidationError("Model with id=\"" + modelId + "\" already exists");
            return exception;
        }

        // Check if modelId is in model graveyard
        // ModelId is added to model graveyard if that model is undergoing deletion
        // and will be removed from it after model is deleted
        if (modelDao.isModelInGraveyard(modelId)) {
            exception = new ActionRequestValidationException();
            String errorMessage = String.format(
                "Model with id = \"%s\" is being deleted. Cannot create a model with same modelID until that model is deleted",
                modelId
            );
            exception.addValidationError(errorMessage);
            return exception;
        }

        // Confirm that the passed in knnMethodContext is valid and requires training
        ValidationException validationException = this.knnMethodContext.validate();
        if (validationException != null) {
            exception = new ActionRequestValidationException();
            exception.addValidationErrors(validationException.validationErrors());
        }

        validationException = this.knnMethodContext.validateWithData(new VectorSpaceInfo(dimension));
        if (validationException != null) {
            exception = new ActionRequestValidationException();
            exception.addValidationErrors(validationException.validationErrors());
        }

        if (!this.knnMethodContext.isTrainingRequired()) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Method does not require training.");
        }

        // Check if preferred node is real
        if (preferredNodeId != null && !clusterService.state().nodes().getDataNodes().containsKey(preferredNodeId)) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Preferred node \"" + preferredNodeId + "\" does not exist");
        }

        // Check if description is too long
        if (description != null && description.length() > KNNConstants.MAX_MODEL_DESCRIPTION_LENGTH) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Description exceeds limit of " + KNNConstants.MAX_MODEL_DESCRIPTION_LENGTH + " characters");
        }

        // Validate training index exists
        IndexMetadata indexMetadata = clusterService.state().metadata().index(trainingIndex);
        if (indexMetadata == null) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Index \"" + this.trainingIndex + "\" does not exist.");
            return exception;
        }

        // Validate the training field
        ValidationException fieldValidation = IndexUtil.validateKnnField(indexMetadata, this.trainingField, this.dimension, modelDao);
        if (fieldValidation != null) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationErrors(fieldValidation.validationErrors());
        }

        return exception;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalString(this.modelId);
        knnMethodContext.writeTo(out);
        out.writeString(this.trainingIndex);
        out.writeString(this.trainingField);
        out.writeOptionalString(this.preferredNodeId);
        out.writeInt(this.dimension);
        out.writeOptionalString(this.description);
        out.writeInt(this.maximumVectorCount);
        out.writeInt(this.searchSize);
        out.writeInt(this.trainingDataSizeInKB);
    }
}
