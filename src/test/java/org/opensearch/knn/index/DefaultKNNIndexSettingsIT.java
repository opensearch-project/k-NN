/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.search.processor.KNNSearchPipelineInitializer.KNN_DEFAULT_SEARCH_PIPELINE_NAME;

public class DefaultKNNIndexSettingsIT extends KNNRestTestCase {

    private static final String DEFAULT_PIPELINE_SETTING = "index.search.default_pipeline";

    public void testNonKnnIndexDoesNotGetDefaultSearchPipeline() throws IOException {
        String indexName = "default-settings-non-knn";
        createIndex(indexName, Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).build());

        assertNull(getIndexSettingByName(indexName, DEFAULT_PIPELINE_SETTING));

        deleteKNNIndex(indexName);
    }

    public void testUserCanOverrideDefaultSearchPipeline() throws IOException {
        String indexName = "default-settings-override";
        createKnnIndex(indexName, createKnnIndexMapping(FIELD_NAME, 2));

        assertEquals(KNN_DEFAULT_SEARCH_PIPELINE_NAME, getIndexSettingByName(indexName, DEFAULT_PIPELINE_SETTING));

        updateIndexSettings(indexName, Settings.builder().put(DEFAULT_PIPELINE_SETTING, "_none"));
        assertEquals("_none", getIndexSettingByName(indexName, DEFAULT_PIPELINE_SETTING));

        deleteKNNIndex(indexName);
    }

    public void testUserSpecifiedPipelineNotOverridden() throws IOException {
        String indexName = "default-settings-user-pipeline";
        String userPipeline = "my_custom_pipeline";
        Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DEFAULT_PIPELINE_SETTING, userPipeline)
            .build();
        createKnnIndex(indexName, settings, createKnnIndexMapping(FIELD_NAME, 2));

        assertEquals(userPipeline, getIndexSettingByName(indexName, DEFAULT_PIPELINE_SETTING));

        deleteKNNIndex(indexName);
    }

    @SuppressWarnings("unchecked")
    public void testSearchResponseExcludesVectorFields() throws Exception {
        String indexName = "default-settings-excludes";
        createKnnIndex(indexName, createKnnIndexMapping(FIELD_NAME, 2));
        addKnnDocWithAttributes(indexName, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f }, Map.of(FIELD_NAME_NON_KNN, "test_value"));
        assertEquals(KNN_DEFAULT_SEARCH_PIPELINE_NAME, getIndexSettingByName(indexName, DEFAULT_PIPELINE_SETTING));

        float[] queryVector = { 1.0f, 2.0f };
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .array("vector", queryVector)
            .field("k", 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Response response = searchKNNIndex(indexName, queryBuilder, 1);
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> source = getFirstHitSource(responseBody);
        assertFalse("Vector field should be excluded from _source", source.containsKey(FIELD_NAME));
        assertEquals("Non-KNN field should be present in _source", "test_value", source.get(FIELD_NAME_NON_KNN));

        deleteKNNIndex(indexName);
    }

    @SuppressWarnings("unchecked")
    public void testSearchResponseIncludesVectorFieldWhenExplicitlyRequested() throws Exception {
        String indexName = "default-settings-explicit-include";
        createKnnIndex(indexName, createKnnIndexMapping(FIELD_NAME, 2));
        addKnnDocWithAttributes(indexName, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f }, Map.of(FIELD_NAME_NON_KNN, "test_value"));

        String query = String.format(
            "{\"_source\": {\"includes\": [\"%s\"]}, \"query\": {\"knn\": {\"%s\": {\"vector\": [1.0, 2.0], \"k\": 1}}}}",
            FIELD_NAME,
            FIELD_NAME
        );
        Response response = searchKNNIndex(indexName, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> source = getFirstHitSource(responseBody);
        assertEquals(List.of(1.0, 2.0), source.get(FIELD_NAME));

        deleteKNNIndex(indexName);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getFirstHitSource(String responseBody) throws IOException {
        List<Map<String, Object>> hits = (List<Map<String, Object>>) ((Map<String, Object>) createParser(
            org.opensearch.core.xcontent.MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits")).get("hits");
        assertEquals(1, hits.size());
        Map<String, Object> source = (Map<String, Object>) hits.get(0).get("_source");
        assertNotNull(source);
        return source;
    }
}
