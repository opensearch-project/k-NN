/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.MediaTypeRegistry;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

/**
 * Rolling-upgrade BWC test for the default KNN search pipeline that auto-excludes vector fields from _source.
 *
 * OLD: Index created without pipeline, vectors in _source.
 * MIXED: Old index still works, vectors still in _source (pipeline not retroactive).
 * UPGRADED: Old index unchanged. New index gets pipeline, vectors excluded.
 */
public class DefaultExcludesBWCIT extends AbstractRollingUpgradeTestCase {

    private static final String TEST_FIELD = "test_vector";
    private static final String NON_KNN_FIELD = "color";
    private static final int DIMENSION = 5;
    private static final int K = 1;
    private static final String DEFAULT_PIPELINE_SETTING = "index.search.default_pipeline";

    public void testDefaultExcludesPipelineRollingUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSION));
                addKnnDocWithAttributes(
                    testIndex,
                    "0",
                    TEST_FIELD,
                    new Float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
                    Map.of(NON_KNN_FIELD, "red")
                );

                assertNull(getIndexSettingByName(testIndex, DEFAULT_PIPELINE_SETTING));

                Map<String, Object> oldSource = searchAndGetFirstSource(testIndex);
                assertTrue("Vector should be in _source on old cluster", oldSource.containsKey(TEST_FIELD));
                assertEquals("red", oldSource.get(NON_KNN_FIELD));
                break;

            case MIXED:
                // Old index still works in mixed cluster, no pipeline retroactively applied
                assertNull(getIndexSettingByName(testIndex, DEFAULT_PIPELINE_SETTING));

                Map<String, Object> mixedSource = searchAndGetFirstSource(testIndex);
                assertTrue("Vector should be in _source in mixed cluster", mixedSource.containsKey(TEST_FIELD));
                assertEquals("red", mixedSource.get(NON_KNN_FIELD));
                break;

            case UPGRADED:
                // Old index still has no pipeline
                assertNull(getIndexSettingByName(testIndex, DEFAULT_PIPELINE_SETTING));

                Map<String, Object> upgradedOldSource = searchAndGetFirstSource(testIndex);
                assertTrue("Old index should still return vectors after upgrade", upgradedOldSource.containsKey(TEST_FIELD));

                // New index created after full upgrade gets the pipeline
                String newIndex = testIndex + "-new";
                createKnnIndex(newIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSION));
                addKnnDocWithAttributes(
                    newIndex,
                    "0",
                    TEST_FIELD,
                    new Float[] { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f },
                    Map.of(NON_KNN_FIELD, "blue")
                );

                assertNotNull(getIndexSettingByName(newIndex, DEFAULT_PIPELINE_SETTING));

                Map<String, Object> newSource = searchAndGetFirstSource(newIndex);
                assertFalse("Vector should be excluded from _source on new index", newSource.containsKey(TEST_FIELD));
                assertEquals("blue", newSource.get(NON_KNN_FIELD));

                deleteKNNIndex(newIndex);
                deleteKNNIndex(testIndex);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> searchAndGetFirstSource(String index) throws Exception {
        String body = String.format(
            "{\"query\": {\"knn\": {\"%s\": {\"vector\": [1.0, 1.0, 1.0, 1.0, 1.0], \"k\": %d}}}}",
            TEST_FIELD,
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
