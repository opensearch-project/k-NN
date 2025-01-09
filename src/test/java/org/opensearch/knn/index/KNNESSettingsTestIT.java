/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;

public class KNNESSettingsTestIT extends KNNRestTestCase {

    public static final int ALWAYS_BUILD_GRAPH = 0;

    /**
     * KNN Index writes should be blocked when the plugin disabled
     * @throws Exception Exception from test
     */
    public void testIndexWritesPluginDisabled() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        float[] qvector = { 1.0f, 2.0f };
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);
        assertEquals("knn query failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        // disable plugin
        updateClusterSettings(KNNSettings.KNN_PLUGIN_ENABLED, false);

        // indexing should be blocked
        Exception ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector));
        assertThat(ex.getMessage(), containsString("KNN plugin is disabled"));

        // enable plugin
        updateClusterSettings(KNNSettings.KNN_PLUGIN_ENABLED, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector);
    }

    public void testQueriesPluginDisabled() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        float[] qvector = { 1.0f, 2.0f };
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);
        assertEquals("knn query failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        // update settings
        updateClusterSettings(KNNSettings.KNN_PLUGIN_ENABLED, false);

        // indexing should be blocked
        Exception ex = expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1)
        );
        assertThat(ex.getMessage(), containsString("KNN plugin is disabled"));
        // enable plugin
        updateClusterSettings(KNNSettings.KNN_PLUGIN_ENABLED, true);
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);
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
