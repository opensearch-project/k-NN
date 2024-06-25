/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlock;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.block.ClusterBlocks;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.IndexService;
import org.opensearch.plugins.Plugin;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.test.OpenSearchSingleNodeTestCase;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.Base64;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class KNNSingleNodeTestCase extends OpenSearchSingleNodeTestCase {
    @Override
    public void setUp() throws Exception {
        super.setUp();
        // Reset all of the counters
        for (KNNCounter knnCounter : KNNCounter.values()) {
            knnCounter.set(0L);
        }
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }

    @Override
    protected boolean resetNodeAfterTest() {
        return true;
    }

    @Override
    public void tearDown() throws Exception {
        NativeMemoryCacheManager.getInstance().invalidateAll();
        NativeMemoryCacheManager.getInstance().close();
        NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance().close();
        NativeMemoryLoadStrategy.TrainingLoadStrategy.getInstance().close();
        NativeMemoryLoadStrategy.AnonymousLoadStrategy.getInstance().close();
        super.tearDown();
    }

    /**
     * Create a k-NN index with default settings
     */
    protected IndexService createKNNIndex(String indexName) {
        return createIndex(indexName, getKNNDefaultIndexSettings());
    }

    /**
     * Create simple k-NN mapping
     */
    protected void createKnnIndexMapping(String indexName, String fieldName, Integer dimensions) {
        PutMappingRequest request = new PutMappingRequest(indexName);
        request.source(fieldName, "type=knn_vector,dimension=" + dimensions);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    /**
     * Create simple k-NN mapping which can be nested.
     * e.g. fieldPath = "a.b.c" will create mapping for "c" as knn_vector
     */
    protected void createKnnNestedIndexMapping(String indexName, String fieldPath, Integer dimensions) throws IOException {
        PutMappingRequest request = new PutMappingRequest(indexName);
        String[] path = fieldPath.split("\\.");
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().startObject("properties");
        for (int i = 0; i < path.length; i++) {
            xContentBuilder.startObject(path[i]);
            if (i == path.length - 1) {
                xContentBuilder.field("type", "knn_vector").field("dimension", dimensions.toString());
            } else {
                xContentBuilder.startObject("properties");
            }
        }
        for (int i = path.length - 1; i >= 0; i--) {
            if (i != path.length - 1) {
                xContentBuilder.endObject();
            }
            xContentBuilder.endObject();
        }
        xContentBuilder.endObject().endObject();

        request.source(xContentBuilder);

        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    /**
     * Get default k-NN settings for test cases
     */
    protected Settings getKNNDefaultIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();
    }

    /**
     * Add a k-NN doc to an index
     */
    protected void addKnnDoc(String index, String docId, String fieldName, Object[] vector) throws IOException, InterruptedException,
        ExecutionException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, vector).endObject();
        IndexRequest indexRequest = new IndexRequest().index(index)
            .id(docId)
            .source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertEquals(response.status(), RestStatus.CREATED);
    }

    /**
     * Add a k-NN doc to an index with nested knn_vector field
     */
    protected void addKnnNestedDoc(String index, String docId, String fieldPath, Object[] vector) throws IOException, InterruptedException,
        ExecutionException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        String[] fieldParts = fieldPath.split("\\.");

        for (int i = 0; i < fieldParts.length - 1; i++) {
            builder.startObject(fieldParts[i]);
        }
        builder.field(fieldParts[fieldParts.length - 1], vector);
        for (int i = fieldParts.length - 2; i >= 0; i--) {
            builder.endObject();
        }
        builder.endObject();
        IndexRequest indexRequest = new IndexRequest().index(index)
            .id(docId)
            .source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertEquals(response.status(), RestStatus.CREATED);
    }

    /**
     * Add any document to index
     */
    protected void addDoc(String index, String docId, String fieldName, String dummyValue) throws IOException, InterruptedException,
        ExecutionException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, dummyValue).endObject();
        IndexRequest indexRequest = new IndexRequest().index(index)
            .id(docId)
            .source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertEquals(response.status(), RestStatus.CREATED);
    }

    /**
     * Index a new model
     */
    protected void writeModelToModelSystemIndex(Model model) throws IOException, ExecutionException, InterruptedException {
        ModelMetadata modelMetadata = model.getModelMetadata();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(MODEL_ID, model.getModelID())
            .field(KNN_ENGINE, modelMetadata.getKnnEngine().getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, modelMetadata.getSpaceType().getValue())
            .field(DIMENSION, modelMetadata.getDimension())
            .field(MODEL_STATE, modelMetadata.getState().getName())
            .field(MODEL_TIMESTAMP, modelMetadata.getTimestamp().toString())
            .field(MODEL_DESCRIPTION, modelMetadata.getDescription())
            .field(MODEL_ERROR, modelMetadata.getError())
            .field(VECTOR_DATA_TYPE_FIELD, modelMetadata.getVectorDataType().getValue());

        if (model.getModelBlob() != null) {
            builder.field(MODEL_BLOB_PARAMETER, Base64.getEncoder().encodeToString(model.getModelBlob()));
        }

        builder.endObject();

        IndexRequest indexRequest = new IndexRequest().index(MODEL_INDEX_NAME)
            .id(model.getModelID())
            .source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(indexRequest).get();
        assertTrue(response.status() == RestStatus.CREATED || response.status() == RestStatus.OK);
    }

    // Add a new model to ModelDao
    protected void addModel(Model model) throws IOException {
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        modelDao.put(model, new ActionListener<IndexResponse>() {
            @Override
            public void onResponse(IndexResponse indexResponse) {
                assertTrue(indexResponse.status() == RestStatus.CREATED || indexResponse.status() == RestStatus.OK);
            }

            @Override
            public void onFailure(Exception e) {
                fail("Failed to add model: " + e);
            }
        });
    }

    /**
     * Run a search against a k-NN index
     */
    protected void searchKNNIndex(String index, String fieldName, float[] vector, int k) {
        SearchResponse response = client().prepareSearch(index).setQuery(new KNNQueryBuilder(fieldName, vector, k)).get();
        assertEquals(response.status(), RestStatus.OK);
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }

    public void assertTrainingSucceeds(ModelDao modelDao, String modelId, int attempts, int delayInMillis) throws InterruptedException,
        ExecutionException {

        int attemptNum = 0;
        ModelMetadata modelMetadata;
        while (attemptNum < attempts) {
            Thread.sleep(delayInMillis);
            attemptNum++;

            if (!modelDao.isCreated()) {
                continue;
            }

            modelMetadata = modelDao.get(modelId).getModelMetadata();

            if (modelMetadata.getState() == ModelState.CREATED) {
                return;
            }

            assertNotEquals(ModelState.FAILED, modelMetadata.getState());
        }

        fail("Training did not succeed after " + attempts + " attempts with a delay of " + delayInMillis + " ms.");
    }

    // Add Global Cluster Block with the given ClusterBlockLevel
    protected void addGlobalClusterBlock(ClusterService clusterService, String description, EnumSet<ClusterBlockLevel> clusterBlockLevels) {
        ClusterBlock block = new ClusterBlock(randomInt(), description, false, false, false, RestStatus.FORBIDDEN, clusterBlockLevels);
        ClusterBlocks clusterBlocks = ClusterBlocks.builder().addGlobalBlock(block).build();
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).blocks(clusterBlocks).build();
        when(clusterService.state()).thenReturn(state);
    }

    // Add Cluster Block for an Index with given ClusterBlockLevel
    protected void addIndexClusterBlock(
        ClusterService clusterService,
        String description,
        EnumSet<ClusterBlockLevel> clusterBlockLevels,
        String testIndex
    ) {
        ClusterBlock block = new ClusterBlock(randomInt(), description, false, false, false, RestStatus.FORBIDDEN, clusterBlockLevels);
        ClusterBlocks clusterBlocks = ClusterBlocks.builder().addIndexBlock(testIndex, block).build();
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).blocks(clusterBlocks).build();
        when(clusterService.state()).thenReturn(state);
    }
}
