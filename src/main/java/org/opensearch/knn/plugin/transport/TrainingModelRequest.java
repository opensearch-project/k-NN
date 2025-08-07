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

import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;

/**
 * Request to train and serialize a model
 */
@Getter
public class TrainingModelRequest extends ActionRequest {

    private static ClusterService clusterService;
    private static ModelDao modelDao;

    private final String modelId;
    private final KNNMethodContext knnMethodContext;
    private final KNNMethodConfigContext knnMethodConfigContext;
    private final int dimension;
    private final String trainingIndex;
    private final String trainingField;
    private final String preferredNodeId;
    private final String description;
    private final VectorDataType vectorDataType;
    private int maximumVectorCount;
    private int searchSize;
    private int trainingDataSizeInKB;
    private final Mode mode;
    private final CompressionLevel compressionLevel;

    TrainingModelRequest(
        String modelId,
        KNNMethodContext knnMethodContext,
        int dimension,
        String trainingIndex,
        String trainingField,
        String preferredNodeId,
        String description,
        VectorDataType vectorDataType,
        Mode mode,
        CompressionLevel compressionLevel
    ) {
        this(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            preferredNodeId,
            description,
            vectorDataType,
            mode,
            compressionLevel,
            SpaceType.DEFAULT,
            knnMethodContext == null ? KNNEngine.DEFAULT : knnMethodContext.getKnnEngine()
        );
    }

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
        String description,
        VectorDataType vectorDataType,
        Mode mode,
        CompressionLevel compressionLevel,
        SpaceType spaceType,
        KNNEngine knnEngine
    ) {
        super();
        this.modelId = modelId;
        this.dimension = dimension;
        this.trainingIndex = trainingIndex;
        this.trainingField = trainingField;
        this.preferredNodeId = preferredNodeId;
        this.description = description;
        this.vectorDataType = vectorDataType;
        this.mode = mode;

        // Set these as defaults initially. If call wants to override them, they can use the setters.
        this.maximumVectorCount = Integer.MAX_VALUE; // By default, get all vectors in the index
        this.searchSize = 10_000; // By default, use the maximum search size

        // Training data size in kilobytes. By default, this is invalid (it cant have negative kb). It eventually gets
        // calculated in transit. A user cannot set this value directly.
        this.knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(vectorDataType)
            .dimension(dimension)
            .versionCreated(Version.CURRENT)
            .compressionLevel(compressionLevel)
            .mode(mode)
            .build();

        ResolvedMethodContext resolvedMethodContext = knnEngine.resolveMethod(knnMethodContext, knnMethodConfigContext, true, spaceType);
        this.knnMethodContext = resolvedMethodContext.getKnnMethodContext();
        this.compressionLevel = resolvedMethodContext.getCompressionLevel();
        this.knnMethodConfigContext.setCompressionLevel(resolvedMethodContext.getCompressionLevel());
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
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY)) {
            this.vectorDataType = VectorDataType.get(in.readString());
        } else {
            this.vectorDataType = VectorDataType.DEFAULT;
        }
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE)) {
            this.mode = Mode.fromName(in.readOptionalString());
            this.compressionLevel = CompressionLevel.fromName(in.readOptionalString());
        } else {
            this.mode = Mode.NOT_CONFIGURED;
            this.compressionLevel = CompressionLevel.NOT_CONFIGURED;
        }

        this.knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(vectorDataType)
            .dimension(dimension)
            .versionCreated(in.getVersion())
            .compressionLevel(compressionLevel)
            .mode(mode)
            .build();
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
        ValidationException validationException = this.knnMethodContext.validate(knnMethodConfigContext);
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
        ValidationException fieldValidation = IndexUtil.validateKnnField(
            indexMetadata,
            this.trainingField,
            this.dimension,
            modelDao,
            vectorDataType,
            knnMethodContext
        );
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
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY)) {
            out.writeString(this.vectorDataType.getValue());
        } else {
            out.writeString(VectorDataType.DEFAULT.getValue());
        }
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE)) {
            out.writeOptionalString(mode.getName());
            out.writeOptionalString(compressionLevel.getName());
        }
    }
}
