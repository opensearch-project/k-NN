/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
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
import org.opensearch.rest.RestStatus;
import org.opensearch.test.OpenSearchSingleNodeTestCase;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ExecutionException;

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
}
