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
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.ActionListener;
import org.opensearch.action.DocWriteRequest;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.FailedNodeException;
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
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheRequest;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheResponse;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataRequest;

import java.io.IOException;
import java.net.URL;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;

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
     * Creates model index. It is possible that the 2 threads call this function simulateously. In this case, one
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
     * @throws IOException   thrown on search
     */
    void get(String modelId, ActionListener<GetModelResponse> listener) throws IOException;

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
            CreateIndexRequest request = new CreateIndexRequest(MODEL_INDEX_NAME).mapping(getMapping())
                .settings(
                    Settings.builder()
                        .put("index.hidden", true)
                        .put("index.number_of_shards", this.numberOfShards)
                        .put("index.number_of_replicas", this.numberOfReplicas)
                );
            client.admin().indices().create(request, actionListener);
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

            IndexRequestBuilder indexRequestBuilder = client.prepareIndex(MODEL_INDEX_NAME);

            indexRequestBuilder.setId(model.getModelID());
            indexRequestBuilder.setSource(parameters);

            indexRequestBuilder.setOpType(requestOpType); // Delegate whether this request can update based on opType
            indexRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

            // After metadata update finishes, remove item from every node's cache if necessary. If no model id is
            // passed then nothing needs to be removed from the cache
            ActionListener<IndexResponse> onMetaListener;
            onMetaListener = ActionListener.wrap(
                indexResponse -> client.execute(
                    RemoveModelFromCacheAction.INSTANCE,
                    new RemoveModelFromCacheRequest(model.getModelID()),
                    ActionListener.wrap(removeModelFromCacheResponse -> {
                        if (!removeModelFromCacheResponse.hasFailures()) {
                            listener.onResponse(indexResponse);
                            return;
                        }

                        String failureMessage = buildRemoveModelErrorMessage(removeModelFromCacheResponse);

                        listener.onFailure(new RuntimeException(failureMessage));
                    }, listener::onFailure)
                ),
                listener::onFailure
            );

            // After the model is indexed, update metadata only if the model is in CREATED state
            ActionListener<IndexResponse> onIndexListener;
            if (ModelState.CREATED.equals(model.getModelMetadata().getState())) {
                onIndexListener = getUpdateModelMetadataListener(model.getModelMetadata(), onMetaListener);
            } else {
                onIndexListener = onMetaListener;
            }

            // Create the model index if it does not already exist
            if (!isCreated()) {
                create(
                    ActionListener.wrap(createIndexResponse -> indexRequestBuilder.execute(onIndexListener), onIndexListener::onFailure)
                );
                return;
            }

            indexRequestBuilder.execute(onIndexListener);
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

        @Override
        public Model get(String modelId) throws ExecutionException, InterruptedException {
            /*
                GET /<model_index>/<modelId>?_local
            */
            GetRequestBuilder getRequestBuilder = new GetRequestBuilder(client, GetAction.INSTANCE, MODEL_INDEX_NAME).setId(modelId)
                .setPreference("_local");
            GetResponse getResponse = getRequestBuilder.execute().get();
            Map<String, Object> responseMap = getResponse.getSourceAsMap();
            return Model.getModelFromSourceMap(responseMap);
        }

        /**
         * Get a model from the system index.  Non-blocking.
         *
         * @param modelId  to retrieve
         * @param actionListener handles get model response
         * @throws IOException thrown on search
         */
        @Override
        public void get(String modelId, ActionListener<GetModelResponse> actionListener) throws IOException {
            /*
                GET /<model_index>/<modelId>?_local
            */
            GetRequestBuilder getRequestBuilder = new GetRequestBuilder(client, GetAction.INSTANCE, MODEL_INDEX_NAME).setId(modelId)
                .setPreference("_local");

            getRequestBuilder.execute(ActionListener.wrap(response -> {
                if (response.isSourceEmpty()) {
                    String errorMessage = "Model does not exist";
                    actionListener.onFailure(new ResourceNotFoundException(modelId, errorMessage));
                    return;
                }
                final Map<String, Object> responseMap = response.getSourceAsMap();
                Model model = Model.getModelFromSourceMap(responseMap);
                actionListener.onResponse(new GetModelResponse(model));

            }, actionListener::onFailure));
        }

        /**
         * searches model from the system index.  Non-blocking.
         *
         * @param request  to retrieve
         * @param actionListener handles get model response
         */
        @Override
        public void search(SearchRequest request, ActionListener<SearchResponse> actionListener) {
            request.indices(MODEL_INDEX_NAME);
            client.search(request, actionListener);
        }

        @Override
        public ModelMetadata getMetadata(String modelId) {
            IndexMetadata indexMetadata = clusterService.state().metadata().index(MODEL_INDEX_NAME);

            if (indexMetadata == null) {
                logger.debug("ModelMetadata for model is null. " + MODEL_INDEX_NAME + " index does not exist.");
                return null;
            }

            Map<String, String> models = indexMetadata.getCustomData(MODEL_METADATA_FIELD);
            if (models == null) {
                logger.debug("ModelMetadata for model is null. " + MODEL_INDEX_NAME + "'s custom metadata does not exist.");
                return null;
            }

            String modelMetadata = models.get(modelId);

            if (modelMetadata == null) {
                logger.debug("ModelMetadata for model is null. Model does not exist.");
                return null;
            }

            return ModelMetadata.fromString(modelMetadata);
        }

        private String getMapping() throws IOException {
            URL url = ModelDao.class.getClassLoader().getResource(MODEL_INDEX_MAPPING_PATH);
            if (url == null) {
                throw new IllegalStateException("Unable to retrieve mapping for \"" + MODEL_INDEX_NAME + "\"");
            }

            return Resources.toString(url, Charsets.UTF_8);
        }

        @Override
        public void delete(String modelId, ActionListener<DeleteModelResponse> listener) {
            // If the index is not created, there is no need to delete the model
            if (!isCreated()) {
                logger.error("Cannot delete model. Model index " + MODEL_INDEX_NAME + "does not exist.");
                String errorMessage = "Cannot delete model. Model index does not exist";
                listener.onResponse(new DeleteModelResponse(modelId, "failed", errorMessage));
                return;
            }

            // Setup delete model request
            DeleteRequestBuilder deleteRequestBuilder = new DeleteRequestBuilder(client, DeleteAction.INSTANCE, MODEL_INDEX_NAME);
            deleteRequestBuilder.setId(modelId);
            deleteRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

            // On model deletion from the index, remove the model from all nodes' model cache
            ActionListener<DeleteResponse> onModelDeleteListener = ActionListener.wrap(deleteResponse -> {
                // If model is not deleted, return with error message
                if (deleteResponse.getResult() != DocWriteResponse.Result.DELETED) {
                    String errorMessage = "Model does not exist";
                    listener.onResponse(new DeleteModelResponse(modelId, deleteResponse.getResult().getLowercase(), errorMessage));
                    return;
                }

                // After model is deleted from the index, make sure the model is evicted from every cache in the
                // cluster
                client.execute(
                    RemoveModelFromCacheAction.INSTANCE,
                    new RemoveModelFromCacheRequest(modelId),
                    ActionListener.wrap(removeModelFromCacheResponse -> {

                        if (!removeModelFromCacheResponse.hasFailures()) {
                            listener.onResponse(new DeleteModelResponse(modelId, deleteResponse.getResult().getLowercase(), null));
                            return;
                        }

                        String failureMessage = buildRemoveModelErrorMessage(removeModelFromCacheResponse);

                        listener.onResponse(new DeleteModelResponse(modelId, "failed", failureMessage));

                    }, e -> listener.onResponse(new DeleteModelResponse(modelId, "failed", e.getMessage())))
                );
            }, e -> listener.onResponse(new DeleteModelResponse(modelId, "failed", e.getMessage())));

            // On model metadata removal, delete the model from the index
            ActionListener<AcknowledgedResponse> onMetadataUpdateListener = ActionListener.wrap(
                acknowledgedResponse -> deleteRequestBuilder.execute(onModelDeleteListener),
                listener::onFailure
            );

            // Remove the metadata asynchronously
            client.execute(
                UpdateModelMetadataAction.INSTANCE,
                new UpdateModelMetadataRequest(modelId, true, null),
                onMetadataUpdateListener
            );
        }

        private String buildRemoveModelErrorMessage(RemoveModelFromCacheResponse response) {
            String failureMessage = "Failed to remove model_id from nodes: ";
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
}
