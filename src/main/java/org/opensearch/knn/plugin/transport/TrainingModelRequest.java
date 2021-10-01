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
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.KNNVectorFieldMapper;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;

import java.io.IOException;
import java.util.Map;

//TODO:
// 1. Double check source map when full API is ready

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

    private long trainingDataSizeInKB;

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
    public TrainingModelRequest(String modelId, KNNMethodContext knnMethodContext, int dimension, String trainingIndex,
                                String trainingField, String preferredNodeId, String description) {
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
        this.trainingDataSizeInKB = in.readLong();
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
            throw new IllegalArgumentException("Maximum vector count " + maximumVectorCount + " is invalid. " +
                    "Maximum vector count must be greater than 0");
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
            throw new IllegalArgumentException("Search size " + searchSize + " is invalid. Search size must be " +
                    "between 0 and 10,000");
        }
        this.searchSize = searchSize;
    }

    /**
     * Getter for training data size in kilobytes.
     *
     * @return trainingDataSizeInKB
     */
    public long getTrainingDataSizeInKB() {
        return trainingDataSizeInKB;
    }

    /**
     * Setter for trainingDataSizeInKB. Package private to prevent users from changing this value directly.
     *
     * @param trainingDataSizeInKB to be set.
     */
    void setTrainingDataSizeInKB(long trainingDataSizeInKB) {
        if (trainingDataSizeInKB <= 0) {
            throw new IllegalArgumentException("Training data size " + trainingDataSizeInKB + " is invalid. " +
                    "Training data size must be greater than 0");
        }
        this.trainingDataSizeInKB = trainingDataSizeInKB;
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        // Check if model id exists via model metadata
        if (modelDao.getMetadata(modelId) != null) {
            exception = new ActionRequestValidationException();
            exception.addValidationError("Model with id=\"" + modelId + "\" already exists");
            return exception;
        }

        // Confirm that the passed in knnMethodContext is valid and requires training
        try {
            this.knnMethodContext.validate();
        } catch (ValidationException ve) {
            exception = new ActionRequestValidationException();
            exception.addValidationError(ve.getMessage());
        }

        if (!this.knnMethodContext.isTrainingRequired()) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Method does not require training.");
        }

        // Validate training data
        IndexMetadata indexMetadata = clusterService.state().metadata().index(trainingIndex);
        if (indexMetadata == null) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Index \"" + this.trainingIndex + "\" does not exist.");
        } else {
            exception = validateTrainingField(indexMetadata, exception);
        }

        // Check if preferred node is real
        if (preferredNodeId != null && !clusterService.state().nodes().getDataNodes().containsKey(preferredNodeId)) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Preferred node \"" + preferredNodeId + "\" does not exist");
        }

        // Check if description is too long
        if (description != null && description.length() > KNNConstants.MAX_MODEL_DESCRIPTION_LENGTH) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Description exceeds limit of " + KNNConstants.MAX_MODEL_DESCRIPTION_LENGTH +
                    " characters");
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
        out.writeLong(this.trainingDataSizeInKB);
    }

    @SuppressWarnings("unchecked")
    private ActionRequestValidationException validateTrainingField(IndexMetadata indexMetadata,
                                                                   ActionRequestValidationException exception) {
        // Index metadata should not be null
        Object trainingFieldMapping = indexMetadata.mapping().getSourceAsMap().get(trainingField); //TODO: Double check this when full API is ready

        // Check field existence
        if (trainingFieldMapping == null) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Field \"" + this.trainingField + "\" does not exist.");
            return exception;
        }

        // Check if field is a map. If not, that is a problem
        if (!(trainingFieldMapping instanceof Map)) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Field info for \"" + this.trainingField + "\" is not a map.");
            return exception;
        }

        Map<String, Object> trainingFieldMap = (Map<String, Object>) trainingFieldMapping;

        // Check fields type is knn_vector
        Object type = trainingFieldMap.get("type");

        if (!(type instanceof String) || !KNNVectorFieldMapper.CONTENT_TYPE.equals(type)) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Field \"" + this.trainingField + "\" is not of type "
                    + KNNVectorFieldMapper.CONTENT_TYPE + ".");
            return exception;
        }

        // Check that the dimension of the method passed in matches that of the model
        Object dimension = trainingFieldMap.get(KNNConstants.DIMENSION);

        // If dimension is null, the training index/field could use a model. In this case, we need to get the model id
        // for the index and then fetch its dimension from the models metadata
        if (dimension == null) {
            String modelId = (String) trainingFieldMap.get(KNNConstants.MODEL_ID);

            if (modelId == null) {
                exception = exception == null ? new ActionRequestValidationException() : exception;
                exception.addValidationError("Field \"" + this.trainingField + "\" does not have a dimension set.");
                return exception;
            }

            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            if (modelMetadata == null) {
                exception = exception == null ? new ActionRequestValidationException() : exception;
                exception.addValidationError("Model \"" + modelId + "\" for field \"" + this.trainingField +
                        "\" does not exist.");
                return exception;
            }

            dimension = modelMetadata.getDimension();
            if ((Integer) dimension != this.dimension) {
                exception = exception == null ? new ActionRequestValidationException() : exception;
                exception.addValidationError("Field \"" + this.trainingField + "\" has dimension " + dimension +
                        ", which is different from dimension specified in the training request: " + this.dimension);
                return exception;
            }

            return exception;
        }

        // If the dimension was found in training fields mapping, check that it equals the models proposed dimension.
        if ((Integer) dimension != this.dimension) {
            exception = exception == null ? new ActionRequestValidationException() : exception;
            exception.addValidationError("Field \"" + this.trainingField + "\" has dimension " + dimension +
                    ", which is different from dimension specified in the training request: " + 0);
            return exception;
        }

        return exception;
    }
}
