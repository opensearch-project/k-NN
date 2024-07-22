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

package org.opensearch.knn.indices;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import lombok.SneakyThrows;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchException;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.DocWriteRequest;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.StepListener;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.action.delete.DeleteAction;
import org.opensearch.action.delete.DeleteRequestBuilder;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.get.GetAction;
import org.opensearch.action.get.GetRequestBuilder;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequestBuilder;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.cluster.health.ClusterIndexHealth;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.exception.DeleteModelException;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheRequest;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheResponse;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardAction;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardRequest;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataRequest;

import java.io.IOException;
import java.net.URL;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;

import static java.util.Objects.isNull;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_MAPPING_PATH;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_METADATA_FIELD;
import static org.opensearch.knn.index.KNNSettings.MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING;
import static org.opensearch.knn.index.KNNSettings.MODEL_INDEX_NUMBER_OF_SHARDS_SETTING;

/**
 * ModelDao is used to interface with the model persistence layer
 */
public interface ModelDao {

    /**
     * Creates model index. It is possible that the 2 threads call this function simultaneously. In this case, one
     * thread will throw a ResourceAlreadyExistsException. This should be caught and handled.
     *
     * @param actionListener CreateIndexResponse listener
     * @throws IOException thrown when get mapping fails
     */
    void create(ActionListener<CreateIndexResponse> actionListener) throws IOException;

    /**
     * Checks if the model index exists
     *
     * @return true if the model index exists; false otherwise
     */
    boolean isCreated();

    /**
     * gets model index's health status
     *
     * @return ClusterHealthStatus of model index
     */
    ClusterHealthStatus getHealthStatus();

    /**
     * Put a model into the system index. Non-blocking
     *
     * @param model Model to be indexed
     * @param listener  handles index response
     */
    void put(Model model, ActionListener<IndexResponse> listener) throws IOException;

    /**
     * Update model of model id with new model.
     *
     * @param model new model
     * @param listener handles index response
     */
    void update(Model model, ActionListener<IndexResponse> listener) throws IOException;

    /**
     * Get a model from the system index. Call blocks.
     *
     * @param modelId to retrieve
     * @return model
     * @throws ExecutionException   thrown on search
     * @throws InterruptedException thrown on search
     */
    Model get(String modelId) throws ExecutionException, InterruptedException;

    /**
     * Get a model from the system index.  Non-blocking.
     *
     * @param modelId to retrieve
     * @param listener  handles get model response
     */
    void get(String modelId, ActionListener<GetModelResponse> listener);

    /**
     * searches model from the system index.  Non-blocking.
     *
     * @param request to retrieve
     * @param listener  handles get model response
     * @throws IOException   thrown on search
     */
    void search(SearchRequest request, ActionListener<SearchResponse> listener) throws IOException;

    /**
     * Get metadata for a model. Non-blocking.
     *
     * @param modelId to retrieve
     * @return modelMetadata. If model metadata does not exist, returns null
     */
    ModelMetadata getMetadata(String modelId);

    /**
     * Delete model from index
     *
     * @param modelId  to delete
     * @param listener handles delete response
     */
    void delete(String modelId, ActionListener<DeleteModelResponse> listener);

    /**
     * Check if modelId is in model graveyard or not. Non-blocking.
     * A modelId is added to model graveyard before deleting that
     * model and removed from it after deleting the model
     *
     * @param modelId to retrieve
     * @return true if modelId is in model graveyard, otherwise return false
     */
    boolean isModelInGraveyard(String modelId);

    /**
     * Implementation of ModelDao for k-NN model index
     */
    final class OpenSearchKNNModelDao implements ModelDao {

        public static Logger logger = LogManager.getLogger(ModelDao.class);

        private int numberOfShards;
        private int numberOfReplicas;

        private static OpenSearchKNNModelDao INSTANCE;
        private static Client client;
        private static ClusterService clusterService;
        private static Settings settings;

        /**
         * Make sure we just have one instance of model index
         *
         * @return ModelIndex instance
         */
        public static synchronized OpenSearchKNNModelDao getInstance() {
            if (INSTANCE == null) {
                INSTANCE = new OpenSearchKNNModelDao();
            }
            return INSTANCE;
        }

        public static void initialize(Client client, ClusterService clusterService, Settings settings) {
            OpenSearchKNNModelDao.client = client;
            OpenSearchKNNModelDao.clusterService = clusterService;
            OpenSearchKNNModelDao.settings = settings;
        }

        private OpenSearchKNNModelDao() {
            numberOfShards = MODEL_INDEX_NUMBER_OF_SHARDS_SETTING.get(settings);
            numberOfReplicas = MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING.get(settings);

            clusterService.getClusterSettings().addSettingsUpdateConsumer(MODEL_INDEX_NUMBER_OF_SHARDS_SETTING, it -> numberOfShards = it);
            clusterService.getClusterSettings()
                .addSettingsUpdateConsumer(MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING, it -> numberOfReplicas = it);
        }

        @Override
        public void create(ActionListener<CreateIndexResponse> actionListener) throws IOException {
            if (isCreated()) {
                return;
            }
            runWithStashedThreadContext(() -> {
                CreateIndexRequest request;
                try {
                    request = new CreateIndexRequest(MODEL_INDEX_NAME).mapping(getMapping())
                        .settings(
                            Settings.builder()
                                .put("index.hidden", true)
                                .put("index.number_of_shards", this.numberOfShards)
                                .put("index.number_of_replicas", this.numberOfReplicas)
                        );
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                client.admin().indices().create(request, actionListener);
            });
        }

        @Override
        public boolean isCreated() {
            return clusterService.state().getRoutingTable().hasIndex(MODEL_INDEX_NAME);
        }

        /**
         * gets model index's health status, provided model index is already created
         *
         * @return ClusterHealthStatus of model index
         */
        @Override
        public ClusterHealthStatus getHealthStatus() throws IndexNotFoundException {
            if (!isCreated()) {
                throw new IndexNotFoundException(MODEL_INDEX_NAME);
            }
            ClusterIndexHealth indexHealth = new ClusterIndexHealth(
                clusterService.state().metadata().index(MODEL_INDEX_NAME),
                clusterService.state().getRoutingTable().index(MODEL_INDEX_NAME)
            );
            return indexHealth.getStatus();
        }

        @Override
        public void put(Model model, ActionListener<IndexResponse> listener) throws IOException {
            // Generate random modelId if modelId is null
            putInternal(model, listener, DocWriteRequest.OpType.CREATE);
        }

        @Override
        public void update(Model model, ActionListener<IndexResponse> listener) throws IOException {
            putInternal(model, listener, DocWriteRequest.OpType.INDEX);
        }

        private void putInternal(Model model, ActionListener<IndexResponse> listener, DocWriteRequest.OpType requestOpType)
            throws IOException {

            if (model == null) {
                throw new IllegalArgumentException("Model cannot be null");
            }

            ModelMetadata modelMetadata = model.getModelMetadata();

            Map<String, Object> parameters = new HashMap<String, Object>() {
                {
                    put(KNNConstants.MODEL_ID, model.getModelID());
                    put(KNNConstants.KNN_ENGINE, modelMetadata.getKnnEngine().getName());
                    put(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, modelMetadata.getSpaceType().getValue());
                    put(KNNConstants.DIMENSION, modelMetadata.getDimension());
                    put(KNNConstants.MODEL_STATE, modelMetadata.getState().getName());
                    put(KNNConstants.MODEL_TIMESTAMP, modelMetadata.getTimestamp());
                    put(KNNConstants.MODEL_DESCRIPTION, modelMetadata.getDescription());
                    put(KNNConstants.MODEL_ERROR, modelMetadata.getError());
                    put(KNNConstants.MODEL_NODE_ASSIGNMENT, modelMetadata.getNodeAssignment());
                    put(KNNConstants.VECTOR_DATA_TYPE_FIELD, modelMetadata.getVectorDataType());

                    MethodComponentContext methodComponentContext = modelMetadata.getMethodComponentContext();
                    if (!methodComponentContext.getName().isEmpty()) {
                        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
                        builder = methodComponentContext.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();
                        put(KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT, builder.toString());
                    }
                }
            };

            byte[] modelBlob = model.getModelBlob();

            if (modelBlob == null && ModelState.CREATED.equals(modelMetadata.getState())) {
                throw new IllegalArgumentException("Model binary cannot be null when model state is CREATED");
            }

            // Only add model if it is not null
            if (modelBlob != null) {
                String base64Model = Base64.getEncoder().encodeToString(modelBlob);
                parameters.put(KNNConstants.MODEL_BLOB_PARAMETER, base64Model);
            }

            final IndexRequestBuilder indexRequestBuilder = client.prepareIndex(MODEL_INDEX_NAME);
            indexRequestBuilder.setId(model.getModelID());
            indexRequestBuilder.setSource(parameters);

            indexRequestBuilder.setOpType(requestOpType); // Delegate whether this request can update based on opType
            indexRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

            // After metadata update finishes, remove item from every node's cache if necessary. If no model id is
            // passed then nothing needs to be removed from the cache
            ActionListener<IndexResponse> onMetaListener;
            onMetaListener = ActionListener.wrap(indexResponse -> {
                client.execute(
                    RemoveModelFromCacheAction.INSTANCE,
                    new RemoveModelFromCacheRequest(model.getModelID()),
                    ActionListener.wrap(removeModelFromCacheResponse -> {
                        if (!removeModelFromCacheResponse.hasFailures()) {
                            listener.onResponse(indexResponse);
                            return;
                        }

                        String failureMessage = buildRemoveModelErrorMessage(model.getModelID(), removeModelFromCacheResponse);

                        listener.onFailure(new RuntimeException(failureMessage));
                    }, listener::onFailure)
                );
            }, listener::onFailure);

            ActionListener<IndexResponse> onIndexListener = getUpdateModelMetadataListener(model.getModelMetadata(), onMetaListener);

            // Create the model index if it does not already exist
            Runnable indexModelRunnable = () -> indexRequestBuilder.execute(onIndexListener);
            if (!isCreated()) {
                create(
                    ActionListener.wrap(
                        createIndexResponse -> ModelDao.runWithStashedThreadContext(indexModelRunnable),
                        onIndexListener::onFailure
                    )
                );
                return;
            }

            ModelDao.runWithStashedThreadContext(indexModelRunnable);
        }

        private ActionListener<IndexResponse> getUpdateModelMetadataListener(
            ModelMetadata modelMetadata,
            ActionListener<IndexResponse> listener
        ) {
            return ActionListener.wrap(
                indexResponse -> client.execute(
                    UpdateModelMetadataAction.INSTANCE,
                    new UpdateModelMetadataRequest(indexResponse.getId(), false, modelMetadata),
                    // Here we wrap the IndexResponse listener around an AcknowledgedListener. This allows us
                    // to pass the indexResponse back up.
                    ActionListener.wrap(acknowledgedResponse -> listener.onResponse(indexResponse), listener::onFailure)
                ),
                listener::onFailure
            );
        }

        @SneakyThrows
        @Override
        public Model get(String modelId) {
            /*
                GET /<model_index>/<modelId>?_local
            */
            try {
                return ModelDao.runWithStashedThreadContext(() -> {
                    GetRequestBuilder getRequestBuilder = new GetRequestBuilder(client, GetAction.INSTANCE, MODEL_INDEX_NAME).setId(modelId)
                        .setPreference("_local");
                    GetResponse getResponse;
                    try {
                        getResponse = getRequestBuilder.execute().get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                    Map<String, Object> responseMap = getResponse.getSourceAsMap();
                    return Model.getModelFromSourceMap(responseMap);
                });
            } catch (RuntimeException runtimeException) {
                // we need to use RuntimeException as container for real exception to keep signature
                // of runWithStashedThreadContext generic
                throw runtimeException.getCause();
            }
        }

        /**
         * Get a model from the system index.  Non-blocking.
         *
         * @param modelId  to retrieve
         * @param actionListener handles get model response
         */
        @Override
        public void get(String modelId, ActionListener<GetModelResponse> actionListener) {
            /*
                GET /<model_index>/<modelId>?_local
            */
            ModelDao.runWithStashedThreadContext(() -> {
                GetRequestBuilder getRequestBuilder = new GetRequestBuilder(client, GetAction.INSTANCE, MODEL_INDEX_NAME).setId(modelId)
                    .setPreference("_local");

                getRequestBuilder.execute(ActionListener.wrap(response -> {
                    if (response.isSourceEmpty()) {
                        String errorMessage = String.format("Model \" %s \" does not exist", modelId);
                        actionListener.onFailure(new ResourceNotFoundException(modelId, errorMessage));
                        return;
                    }
                    final Map<String, Object> responseMap = response.getSourceAsMap();
                    Model model = Model.getModelFromSourceMap(responseMap);
                    actionListener.onResponse(new GetModelResponse(model));

                }, actionListener::onFailure));
            });
        }

        /**
         * searches model from the system index.  Non-blocking.
         *
         * @param request  to retrieve
         * @param actionListener handles get model response
         */
        @Override
        public void search(SearchRequest request, ActionListener<SearchResponse> actionListener) {
            ModelDao.runWithStashedThreadContext(() -> {
                request.indices(MODEL_INDEX_NAME);
                client.search(request, actionListener);
            });
        }

        @Override
        public ModelMetadata getMetadata(String modelId) {
            IndexMetadata indexMetadata = clusterService.state().metadata().index(MODEL_INDEX_NAME);

            if (indexMetadata == null) {
                logger.debug("ModelMetadata for model \"" + modelId + "\" is null. " + MODEL_INDEX_NAME + " index does not exist.");
                return null;
            }

            Map<String, String> models = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
            if (models == null) {
                logger.debug(
                    "ModelMetadata for model \"" + modelId + "\" is null. " + MODEL_INDEX_NAME + "'s custom metadata does not exist."
                );
                return null;
            }

            String modelMetadata = models.get(modelId);

            if (modelMetadata == null) {
                logger.debug("ModelMetadata for model \"" + modelId + "\" is null. Model \"" + modelId + "\" does " + "not exist.");
                return null;
            }

            return ModelMetadata.fromString(modelMetadata);
        }

        private String getMapping() throws IOException {
            if (ModelDao.class.getClassLoader() == null) {
                throw new IllegalStateException("ClassLoader of ModelDao Class is null");
            }
            URL url = ModelDao.class.getClassLoader().getResource(MODEL_INDEX_MAPPING_PATH);
            if (url == null) {
                throw new IllegalStateException("Unable to retrieve mapping for \"" + MODEL_INDEX_NAME + "\"");
            }

            return Resources.toString(url, Charsets.UTF_8);
        }

        @Override
        public boolean isModelInGraveyard(String modelId) {
            // Check if the objects are not null and throw a customized NullPointerException
            Objects.requireNonNull(clusterService.state(), "Cluster state must not be null");
            Objects.requireNonNull(clusterService.state().metadata(), "Cluster metadata must not be null");
            ModelGraveyard modelGraveyard = clusterService.state().metadata().custom(ModelGraveyard.TYPE);

            if (isNull(modelGraveyard)) {
                return false;
            }

            return modelGraveyard.contains(modelId);
        }

        @Override
        public void delete(String modelId, ActionListener<DeleteModelResponse> listener) {
            // If the index is not created, there is no need to delete the model
            if (!isCreated()) {
                String errorMessage = String.format("Cannot delete model [%s]. Model index [%s] does not exist", modelId, MODEL_INDEX_NAME);
                listener.onFailure(new ResourceNotFoundException(errorMessage));
                return;
            }

            StepListener<GetModelResponse> getModelStep = new StepListener<>();
            StepListener<AcknowledgedResponse> blockModelIdStep = new StepListener<>();
            StepListener<AcknowledgedResponse> clearModelMetadataStep = new StepListener<>();
            StepListener<DeleteResponse> deleteModelFromIndexStep = new StepListener<>();
            StepListener<RemoveModelFromCacheResponse> clearModelFromCacheStep = new StepListener<>();
            StepListener<AcknowledgedResponse> unblockModelIdStep = new StepListener<>();

            // Get Model to check if model is in TRAINING
            get(modelId, ActionListener.wrap(getModelStep::onResponse, exception -> {
                if (exception instanceof ResourceNotFoundException) {
                    String errorMessage = String.format("Unable to delete model [%s]. Model does not exist", modelId);
                    ResourceNotFoundException resourceNotFoundException = new ResourceNotFoundException(errorMessage);
                    removeModelIdFromGraveyardOnFailure(modelId, resourceNotFoundException, getModelStep);
                } else {
                    removeModelIdFromGraveyardOnFailure(modelId, exception, getModelStep);
                }
            }));

            getModelStep.whenComplete(getModelResponse -> {
                // If model is in Training state, fail delete model request
                if (ModelState.TRAINING == getModelResponse.getModel().getModelMetadata().getState()) {
                    String errorMessage = String.format("Cannot delete model [%s]. Model is still in training", modelId);
                    listener.onFailure(new DeleteModelException(errorMessage));
                    return;
                }

                // Add modelId to model graveyard until delete model request is processed
                updateModelGraveyardToDelete(modelId, false, blockModelIdStep, Optional.empty());
            }, listener::onFailure);

            // Remove the metadata asynchronously
            blockModelIdStep.whenComplete(
                acknowledgedResponse -> { clearModelMetadata(modelId, clearModelMetadataStep); },
                listener::onFailure
            );

            // Setup delete model request
            clearModelMetadataStep.whenComplete(acknowledgedResponse -> {
                DeleteRequestBuilder deleteRequestBuilder = new DeleteRequestBuilder(client, DeleteAction.INSTANCE, MODEL_INDEX_NAME);
                deleteRequestBuilder.setId(modelId);
                deleteRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                deleteModelFromIndex(modelId, deleteModelFromIndexStep, deleteRequestBuilder);
            }, listener::onFailure);

            deleteModelFromIndexStep.whenComplete(deleteResponse -> {
                // If model is not deleted, remove modelId from model graveyard and return with error message
                if (deleteResponse.getResult() != DocWriteResponse.Result.DELETED) {
                    updateModelGraveyardToDelete(modelId, true, unblockModelIdStep, Optional.empty());
                    String errorMessage = String.format("Model [%s] does not exist", modelId);
                    listener.onFailure(new ResourceNotFoundException(errorMessage));
                    return;
                }

                // After model is deleted from the index, make sure the model is evicted from every cache in the cluster
                removeModelFromCache(modelId, clearModelFromCacheStep);
            }, e -> listener.onFailure(new OpenSearchException(e)));

            clearModelFromCacheStep.whenComplete(removeModelFromCacheResponse -> {

                // If there are any failures while removing model from the cache build the error message
                OpenSearchException exception = null;
                if (removeModelFromCacheResponse.hasFailures()) {
                    String failureMessage = buildRemoveModelErrorMessage(modelId, removeModelFromCacheResponse);
                    exception = new OpenSearchException(failureMessage);
                }

                // Remove modelId from model graveyard
                updateModelGraveyardToDelete(modelId, true, unblockModelIdStep, Optional.ofNullable(exception));

            }, e -> listener.onFailure(new OpenSearchException(e)));

            unblockModelIdStep.whenComplete(acknowledgedResponse -> {
                // After clearing the cache, if there are no errors return the response
                listener.onResponse(new DeleteModelResponse(modelId));

            }, listener::onFailure);

        }

        // Remove model from cache in the cluster
        private void removeModelFromCache(String modelId, StepListener<RemoveModelFromCacheResponse> clearModelFromCacheStep) {
            client.execute(
                RemoveModelFromCacheAction.INSTANCE,
                new RemoveModelFromCacheRequest(modelId),
                ActionListener.wrap(
                    clearModelFromCacheStep::onResponse,
                    exception -> removeModelIdFromGraveyardOnFailure(modelId, exception, clearModelFromCacheStep)
                )
            );
        }

        // Delete model from the system index
        private void deleteModelFromIndex(
            String modelId,
            StepListener<DeleteResponse> deleteModelFromIndexStep,
            DeleteRequestBuilder deleteRequestBuilder
        ) {
            ModelDao.runWithStashedThreadContext(
                () -> deleteRequestBuilder.execute(
                    ActionListener.wrap(
                        deleteModelFromIndexStep::onResponse,
                        exception -> removeModelIdFromGraveyardOnFailure(modelId, exception, deleteModelFromIndexStep)
                    )
                )
            );
        }

        // Update model graveyard to add/remove modelId
        private void updateModelGraveyardToDelete(
            String modelId,
            boolean isRemoveRequest,
            StepListener<AcknowledgedResponse> step,
            Optional<Exception> exception
        ) {

            client.execute(
                UpdateModelGraveyardAction.INSTANCE,
                new UpdateModelGraveyardRequest(modelId, isRemoveRequest),
                ActionListener.wrap(acknowledgedResponse -> {
                    if (exception.isEmpty()) {
                        step.onResponse(acknowledgedResponse);
                        return;
                    }
                    throw exception.get();

                }, e -> {
                    // If it fails to remove the modelId from Model Graveyard, then log the error message
                    String errorMessage = String.format("Failed to remove \" %s \" from Model Graveyard", modelId);
                    String failureMessage = String.format("%s%s%s", errorMessage, "\n", e.getMessage());
                    logger.error(failureMessage);

                    if (exception.isEmpty()) {
                        step.onFailure(e);
                        return;
                    }
                    step.onFailure(exception.get());
                })
            );
        }

        // Clear the metadata of the model for a given modelId
        private void clearModelMetadata(String modelId, StepListener<AcknowledgedResponse> clearModelMetadataStep) {
            client.execute(
                UpdateModelMetadataAction.INSTANCE,
                new UpdateModelMetadataRequest(modelId, true, null),
                ActionListener.wrap(
                    clearModelMetadataStep::onResponse,
                    exception -> removeModelIdFromGraveyardOnFailure(modelId, exception, clearModelMetadataStep)
                )
            );
        }

        // This function helps to remove the model from model graveyard and return the exception from previous step
        // when the delete request fails while executing after adding modelId to model graveyard
        private void removeModelIdFromGraveyardOnFailure(String modelId, Exception exceptionFromPreviousStep, StepListener<?> step) {
            client.execute(
                UpdateModelGraveyardAction.INSTANCE,
                new UpdateModelGraveyardRequest(modelId, true),
                ActionListener.wrap(acknowledgedResponse -> {
                    throw exceptionFromPreviousStep;
                }, unblockingFailedException -> {
                    // If it fails to remove the modelId from Model Graveyard, then log the error message and
                    // throw the exception that was passed as a parameter from previous step
                    String errorMessage = String.format("Failed to remove \" %s \" from Model Graveyard", modelId);
                    String failureMessage = String.format("%s%s%s", errorMessage, "\n", unblockingFailedException.getMessage());
                    logger.error(failureMessage);
                    step.onFailure(exceptionFromPreviousStep);
                })
            );
        }

        private String buildRemoveModelErrorMessage(String modelId, RemoveModelFromCacheResponse response) {
            String failureMessage = "Failed to remove \"" + modelId + "\" from nodes: ";
            StringBuilder stringBuilder = new StringBuilder(failureMessage);

            for (FailedNodeException nodeException : response.failures()) {
                stringBuilder.append("Node \"")
                    .append(nodeException.nodeId())
                    .append("\" ")
                    .append(nodeException.getMessage())
                    .append("; ");
            }

            return stringBuilder.toString();
        }
    }

    /**
     * Set the thread context to default, this is needed to allow actions on model system index
     * when security plugin is enabled
     * @param function runnable that needs to be executed after thread context has been stashed, accepts and returns nothing
     */
    private static void runWithStashedThreadContext(Runnable function) {
        try (ThreadContext.StoredContext context = OpenSearchKNNModelDao.client.threadPool().getThreadContext().stashContext()) {
            function.run();
        }
    }

    /**
     * Set the thread context to default, this is needed to allow actions on model system index
     * when security plugin is enabled
     * @param function supplier function that needs to be executed after thread context has been stashed, return object
     */
    private static <T> T runWithStashedThreadContext(Supplier<T> function) {
        try (ThreadContext.StoredContext context = OpenSearchKNNModelDao.client.threadPool().getThreadContext().stashContext()) {
            return function.get();
        }
    }
}
