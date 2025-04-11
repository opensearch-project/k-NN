/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD;

public class KNNESSettingsTestIT extends KNNRestTestCase {

    public static final int ALWAYS_BUILD_GRAPH = 0;

    public void testKNNLegacySpaceTypeIndexingTest() throws IOException, ParseException {
        // Configure space_type at index level. This is deprecated and will be removed in the future.
        Settings.Builder builder = Settings.builder().put("index.knn", true).put("knn.algo_param.ef_search", 100);

        final String remoteBuild = System.getProperty("test.remoteBuild", null);
        if (isRemoteIndexBuildSupported(getBWCVersion()) && remoteBuild != null) {
            builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD, true);
            builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "0kb");
        }

        final Settings indexSettings = builder.build();

        // This mapping does not have method.
        final String testField = "knn_field";
        final String testIndex = "knn_index";
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(testField)
            .field(VECTOR_TYPE, KNN_VECTOR)
            .field(DIMENSION, 2)
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(testIndex, indexSettings, mapping);

        // Ingest data.
        float[] queryVector = new float[] { 2, 3 };
        final int k = 2;

        float[][] vectorData = new float[5][2];
        vectorData[0] = new float[] { 11.7f, 2.7f };  // distance=31.5
        vectorData[1] = new float[] { 20.9f, 3.9f };  // distance=53.5 <- answer
        vectorData[2] = new float[] { 3.77f, 4.22f }; // distance=20.2
        vectorData[3] = new float[] { 15, 6 };        // distance=48 <- answer
        vectorData[4] = new float[] { 4.7f, 5.9f };   // distance=27.1

        bulkAddKnnDocs(testIndex, testField, vectorData, vectorData.length);
        flushIndex(testIndex);

        // Send a query and verify scores are correct.
        Response searchResponse = searchKNNIndex(
            testIndex,
            KNNQueryBuilder.builder().k(k).fieldName(testField).vector(queryVector).build(),
            k
        );

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), testField);

        assertEquals(k, results.size());
        Set<String> docIds = new HashSet<>();
        for (KNNResult result : results) {
            docIds.add(result.getDocId());
        }
        assertEquals(new HashSet<>(Arrays.asList("2", "4")), docIds);
    }

    public void testItemRemovedFromCache_expiration() throws Exception {
        createKnnIndex(INDEX_NAME, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(FIELD_NAME, 2));
        updateClusterSettings(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED, true);
        updateClusterSettings(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, "1m");

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        float[] qvector = { 1.0f, 2.0f };
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);
        assertEquals("knn query failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        assertEquals(1, getTotalGraphsInCache());

        Thread.sleep(65 * 1000);

        assertEquals(0, getTotalGraphsInCache());

        updateClusterSettings(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED, false);
    }

    public void testCreateIndexWithInvalidSpaceType() throws IOException {
        String invalidSpaceType = "bar";
        Settings invalidSettings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.space_type", invalidSpaceType)
            .build();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, invalidSettings, createKnnIndexMapping(FIELD_NAME, 2)));
    }

    public void testUpdateIndexSetting() throws IOException {
        Settings settings = Settings.builder()
            .put("index.knn", true)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, 512)
            .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));
        assertEquals("512", getIndexSettingByName(INDEX_NAME, KNNSettings.KNN_ALGO_PARAM_EF_SEARCH));

        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, 400));
        assertEquals("400", getIndexSettingByName(INDEX_NAME, KNNSettings.KNN_ALGO_PARAM_EF_SEARCH));

        Exception ex = expectThrows(
            ResponseException.class,
            () -> updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, 1))
        );
        assertThat(ex.getMessage(), containsString("Failed to parse value [1] for setting [index.knn.algo_param.ef_search] must be >= 2"));
    }

    public void testUpdateIndexSettingKnnFlagImmutable() throws IOException {
        Settings settings = Settings.builder().put(KNNSettings.KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));

        Exception ex = expectThrows(
            ResponseException.class,
            () -> updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_INDEX, false))
        );
        assertThat(ex.getMessage(), containsString("Can't update non dynamic settings [[index.knn]] for open indices"));

        closeIndex(INDEX_NAME);

        ex = expectThrows(
            ResponseException.class,
            () -> updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_INDEX, false))
        );
        assertThat(ex.getMessage(), containsString(String.format("final %s setting [index.knn], not updateable", INDEX_NAME)));

    }

    @SuppressWarnings("unchecked")
    public void testCacheRebuiltAfterUpdateIndexSettings() throws Exception {
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        float[] qvector = { 6.0f, 6.0f };
        // First search to load graph into cache
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats = parseNodeStatsResponse(responseBody).get(0);
        Map<String, Object> indicesInCache = (Map<String, Object>) nodeStats.get(StatNames.INDICES_IN_CACHE.getName());

        assertEquals(1, indicesInCache.size());

        // Update ef_search
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, 400));
        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        responseBody = EntityUtils.toString(response.getEntity());

        nodeStats = parseNodeStatsResponse(responseBody).get(0);
        indicesInCache = (Map<String, Object>) nodeStats.get(StatNames.INDICES_IN_CACHE.getName());

        assertEquals(0, indicesInCache.size());
    }
}
