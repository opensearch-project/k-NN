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
/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */
package com.amazon.opendistroforelasticsearch.knn;

import com.amazon.opendistroforelasticsearch.knn.index.KNNIndexCache;
import com.amazon.opendistroforelasticsearch.knn.index.KNNQueryBuilder;
import com.amazon.opendistroforelasticsearch.knn.plugin.KNNPlugin;
import com.amazon.opendistroforelasticsearch.knn.plugin.stats.KNNCounter;
import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.IndexService;
import org.opensearch.plugins.Plugin;
import org.opensearch.rest.RestStatus;
import org.opensearch.test.OpenSearchSingleNodeTestCase;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
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
        KNNIndexCache.getInstance().evictAllGraphsFromCache();
        KNNIndexCache.getInstance().close();
        super.tearDown();
    }

    /**
     * Create a k-NN index with default settings
     */
    protected IndexService createKNNIndex(String indexName) {
        return createIndex(indexName, getKNNDefaultIndexSettings(), null);
    }

    /**
     * Create simple k-NN mapping
     */
    protected void createKnnIndexMapping(String indexName, String fieldName, Integer dimensions) {
        PutMappingRequest request = new PutMappingRequest(indexName).type("_doc");
        request.source(fieldName, "type=knn_vector,dimension="+dimensions);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    /**
     * Get default k-NN settings for test cases
     */
    protected Settings getKNNDefaultIndexSettings() {
        return Settings.builder()
                .put("number_of_shards", 1)
                .put("number_of_replicas", 0)
                .put("index.knn", true)
                .build();
    }

    /**
     * Add a k-NN doc to an index
     */
    protected void addKnnDoc(String index, String docId, String fieldName, Object[] vector)
            throws IOException, InterruptedException, ExecutionException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
                .field(fieldName, vector)
                .endObject();
        IndexRequest indexRequest = new IndexRequest()
                .index(index)
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
        SearchResponse response = client().prepareSearch(index).setQuery(new KNNQueryBuilder(fieldName, vector, k))
                .get();
        assertEquals(response.status(), RestStatus.OK);
    }
}
