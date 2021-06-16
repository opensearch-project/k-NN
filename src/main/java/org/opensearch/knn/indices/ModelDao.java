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
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Resources;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.ActionListener;
import org.opensearch.action.DocWriteRequest;
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
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.net.URL;
import java.util.Base64;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_MAPPING_PATH;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
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
     * Put a model into the system index. Non-blocking
     *
     * @param modelId   Id of model to create
     * @param modelBlob byte array of model
     * @param listener  handles index response
     */
    void put(String modelId, KNNEngine knnEngine, byte[] modelBlob, ActionListener<IndexResponse> listener);

    /**
     * Put a model into the system index. Non-blocking. When no id is passed in, OpenSearch will generate the id
     * automatically. The id can be retrieved in the IndexResponse.
     *
     * @param modelBlob byte array of model
     * @param listener  handles index response
     */
    void put(KNNEngine knnEngine, byte[] modelBlob, ActionListener<IndexResponse> listener);

    /**
     * Get a model from the system index. Call blocks.
     *
     * @param modelId to retrieve
     * @return byte array representing the model
     * @throws ExecutionException   thrown on search
     * @throws InterruptedException thrown on search
     */
    byte[] get(String modelId) throws ExecutionException, InterruptedException;

    /**
     * Delete model from index
     *
     * @param modelId  to delete
     * @param listener handles delete response
     */
    void delete(String modelId, ActionListener<DeleteResponse> listener);

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

            clusterService.getClusterSettings().addSettingsUpdateConsumer(MODEL_INDEX_NUMBER_OF_SHARDS_SETTING,
                    it -> numberOfShards = it);
            clusterService.getClusterSettings().addSettingsUpdateConsumer(MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING,
                    it -> numberOfReplicas = it);
        }

        @Override
        public void create(ActionListener<CreateIndexResponse> actionListener) throws IOException {
            if (isCreated()) {
                return;
            }

            CreateIndexRequest request = new CreateIndexRequest(MODEL_INDEX_NAME)
                    .mapping("_doc", getMapping(), XContentType.JSON)
                    .settings(Settings.builder()
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

        @Override
        public void put(String modelId, KNNEngine knnEngine, byte[] modelBlob, ActionListener<IndexResponse> listener) {
            String base64Model = Base64.getEncoder().encodeToString(modelBlob);

            Map<String, Object> parameters = ImmutableMap.of(
                    KNNConstants.KNN_ENGINE, knnEngine.getName(),
                    KNNConstants.MODEL_BLOB_PARAMETER, base64Model
            );

            IndexRequestBuilder indexRequestBuilder = client.prepareIndex(MODEL_INDEX_NAME, "_doc");
            indexRequestBuilder.setId(modelId);
            indexRequestBuilder.setSource(parameters);

            put(indexRequestBuilder, listener);
        }

        @Override
        public void put(KNNEngine knnEngine, byte[] modelBlob, ActionListener<IndexResponse> listener) {
            String base64Model = Base64.getEncoder().encodeToString(modelBlob);

            Map<String, Object> parameters = ImmutableMap.of(
                    KNNConstants.KNN_ENGINE, knnEngine.getName(),
                    KNNConstants.MODEL_BLOB_PARAMETER, base64Model
            );

            IndexRequestBuilder indexRequestBuilder = client.prepareIndex(MODEL_INDEX_NAME, "_doc");
            indexRequestBuilder.setSource(parameters);

            put(indexRequestBuilder, listener);
        }

        private void put(IndexRequestBuilder indexRequestBuilder, ActionListener<IndexResponse> listener) {
            if (!isCreated()) {
                throw new IllegalStateException("Cannot put model in index before index has been initialized");
            }

            // Fail if the id already exists. Models are not updateable
            indexRequestBuilder.setOpType(DocWriteRequest.OpType.CREATE);
            indexRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            indexRequestBuilder.execute(listener);
        }

        @Override
        public byte[] get(String modelId) throws ExecutionException, InterruptedException {
            if (!isCreated()) {
                throw new IllegalStateException("Cannot get model \"" + modelId + "\". Model index does not exist.");
            }

            /*
                GET /<model_index>/<modelId>?source_includes=<model_blob>&_local
            */
            GetRequestBuilder getRequestBuilder = new GetRequestBuilder(client, GetAction.INSTANCE, MODEL_INDEX_NAME)
                    .setId(modelId)
                    .setFetchSource(KNNConstants.MODEL_BLOB_PARAMETER, null)
                    .setPreference("_local");
            GetResponse getResponse = getRequestBuilder.execute().get();

            Object blob = getResponse.getSourceAsMap().get(KNNConstants.MODEL_BLOB_PARAMETER);

            if (blob == null) {
                throw new IllegalArgumentException("No model available in \"" + MODEL_INDEX_NAME + "\" index with id \""
                        + modelId + "\".");
            }

            return Base64.getDecoder().decode((String) blob);
        }

        private String getMapping() throws IOException {
            URL url = ModelDao.class.getClassLoader().getResource(MODEL_INDEX_MAPPING_PATH);
            if (url == null) {
                throw new IllegalStateException("Unable to retrieve mapping for \"" + MODEL_INDEX_NAME + "\"");
            }

            return Resources.toString(url, Charsets.UTF_8);
        }

        @Override
        public void delete(String modelId, ActionListener<DeleteResponse> listener) {
            if (!isCreated()) {
                throw new IllegalStateException("Cannot delete model \"" + modelId + "\". Model index does not exist.");
            }

            DeleteRequestBuilder deleteRequestBuilder = new DeleteRequestBuilder(client, DeleteAction.INSTANCE,
                    MODEL_INDEX_NAME);
            deleteRequestBuilder.setId(modelId);
            deleteRequestBuilder.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            deleteRequestBuilder.execute(ActionListener.wrap(deleteResponse -> {
                ModelCache.getInstance().remove(modelId);
                listener.onResponse(deleteResponse);
            }, listener::onFailure));
        }
    }
}
