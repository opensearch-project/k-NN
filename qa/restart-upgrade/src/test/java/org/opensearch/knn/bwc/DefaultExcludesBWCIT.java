/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.MediaTypeRegistry;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

/**
 * BWC test for the default KNN search pipeline that auto-excludes vector fields from _source.
 *
 * Old cluster: no default pipeline exists, indices created without it, vectors returned in _source.
 * Upgraded cluster: pipeline exists, old indices still work (no pipeline), new indices get the pipeline
 * and vectors are excluded from _source by default.
 */
public class DefaultExcludesBWCIT extends AbstractRestartUpgradeTestCase {

    private static final String TEST_FIELD = "test_vector";
    private static final String NON_KNN_FIELD = "color";
    private static final int DIMENSION = 5;
    private static final int K = 1;
    private static final String DEFAULT_PIPELINE_SETTING = "index.search.default_pipeline";

    /**
     * Tests that indices created on old cluster continue to work after upgrade,
     * and new indices created after upgrade get the default excludes pipeline.
     */
    public void testDefaultExcludesPipelineAfterRestart() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            // Create index on old cluster — no default pipeline feature exists
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSION));
            addKnnDocWithAttributes(testIndex, "0", TEST_FIELD, new Float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, Map.of(NON_KNN_FIELD, "red"));

            // Old cluster: no default pipeline setting
            assertNull(
                "Old cluster index should not have default search pipeline",
                getIndexSettingByName(testIndex, DEFAULT_PIPELINE_SETTING)
            );

            // Vectors should be in _source on old cluster
            Map<String, Object> source = searchAndGetFirstSource(testIndex);
            assertTrue("Vector should be in _source on old cluster", source.containsKey(TEST_FIELD));
            assertEquals("red", source.get(NON_KNN_FIELD));
        } else {
            // After upgrade: old index should still work, pipeline not retroactively applied
            assertNull(
                "Old index should not gain default pipeline after upgrade",
                getIndexSettingByName(testIndex, DEFAULT_PIPELINE_SETTING)
            );

            // Search on old index still returns vectors (no pipeline)
            Map<String, Object> oldSource = searchAndGetFirstSource(testIndex);
            assertTrue("Old index should still return vectors after upgrade", oldSource.containsKey(TEST_FIELD));
            assertEquals("red", oldSource.get(NON_KNN_FIELD));

            // Create new index on upgraded cluster — should get default pipeline
            String newIndex = testIndex + "-new";
            createKnnIndex(newIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSION));
            addKnnDocWithAttributes(newIndex, "0", TEST_FIELD, new Float[] { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f }, Map.of(NON_KNN_FIELD, "blue"));

            assertNotNull(
                "New index should have default search pipeline",
                getIndexSettingByName(newIndex, DEFAULT_PIPELINE_SETTING)
            );

            // New index: vectors excluded from _source
            Map<String, Object> newSource = searchAndGetFirstSource(newIndex);
            assertFalse("Vector should be excluded from _source on new index", newSource.containsKey(TEST_FIELD));
            assertEquals("blue", newSource.get(NON_KNN_FIELD));

            deleteKNNIndex(newIndex);
            deleteKNNIndex(testIndex);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> searchAndGetFirstSource(String index) throws Exception {
        float[] queryVector = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        String body = String.format(
            "{\"query\": {\"knn\": {\"%s\": {\"vector\": [%s], \"k\": %d}}}}",
            TEST_FIELD,
            "1.0, 1.0, 1.0, 1.0, 1.0",
            K
        );
        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(body);
        Response response = client().performRequest(request);
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> parsed = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        List<Map<String, Object>> hits = (List<Map<String, Object>>) ((Map<String, Object>) parsed.get("hits")).get("hits");
        assertEquals(1, hits.size());
        Map<String, Object> source = (Map<String, Object>) hits.get(0).get("_source");
        assertNotNull(source);
        return source;
    }
}
