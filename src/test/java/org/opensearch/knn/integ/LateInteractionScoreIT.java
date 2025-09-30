/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.script.Script;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Integration tests for late interaction scoring functionality in k-NN plugin.
 * Tests the lateInteractionScore function in real OpenSearch cluster scenarios.
 */
public class LateInteractionScoreIT extends KNNRestTestCase {

    /**
     * Tests late interaction score calculation with valid vectors.
     * Verifies that the script executes successfully and returns expected results.
     */
    public void testLateInteractionScore_whenValidVectors_thenReturnsCorrectScore() throws Exception {
        createIndexWithMapping();
        indexDocuments();

        // Create query vectors as script parameters
        List<List<Double>> queryVectors = new ArrayList<>();
        List<Double> qv1 = new ArrayList<>();
        qv1.add(0.1);
        qv1.add(0.2);
        queryVectors.add(qv1);

        Map<String, Object> params = new HashMap<>();
        params.put("query_vector", queryVectors);

        String source = "lateInteractionScore(params.query_vector, 'my_vector', params._source)";

        QueryBuilder qb = new MatchAllQueryBuilder();
        Request request = constructScriptScoreContextSearchRequest(
            INDEX_NAME,
            qb,
            params,
            Script.DEFAULT_SCRIPT_LANG,
            source,
            2,
            Collections.emptyMap()
        );

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertTrue("Response should contain hits", responseBody.contains("\"hits\""));
        assertTrue("Response should contain scores", responseBody.contains("\"_score\""));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Tests late interaction score with different space types.
     * Verifies that the function works with various similarity metrics.
     */
    public void testLateInteractionScore_whenDifferentSpaceTypes_thenReturnsCorrectScore() throws Exception {
        createIndexWithMapping();
        indexDocuments();

        List<List<Double>> queryVectors = new ArrayList<>();
        List<Double> qv1 = new ArrayList<>();
        qv1.add(1.0);
        qv1.add(0.0);
        queryVectors.add(qv1);

        String[] spaceTypes = { "innerproduct", "cosinesimil", "l2", "l1", "linf" };

        for (String spaceType : spaceTypes) {
            Map<String, Object> params = new HashMap<>();
            params.put("query_vector", queryVectors);
            params.put("space_type", spaceType);

            String source = "lateInteractionScore(params.query_vector, 'my_vector', params._source, params.space_type)";

            QueryBuilder qb = new MatchAllQueryBuilder();
            Request request = constructScriptScoreContextSearchRequest(
                INDEX_NAME,
                qb,
                params,
                Script.DEFAULT_SCRIPT_LANG,
                source,
                2,
                Collections.emptyMap()
            );

            Response response = client().performRequest(request);
            assertEquals(
                "Space type " + spaceType + " should work",
                RestStatus.OK,
                RestStatus.fromCode(response.getStatusLine().getStatusCode())
            );

            String responseBody = EntityUtils.toString(response.getEntity());
            assertTrue("Response should contain hits for " + spaceType, responseBody.contains("\"hits\""));
        }

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Creates an index with appropriate mapping for vector fields.
     * Sets up the test index with object type field for storing vectors.
     */
    private void createIndexWithMapping() throws Exception {
        String mapping = "{\n"
            + "  \"properties\": {\n"
            + "    \"my_vector\": {\n"
            + "      \"type\": \"object\",\n"
            + "      \"enabled\": false\n"
            + "    }\n"
            + "  }\n"
            + "}";
        createKnnIndex(INDEX_NAME, mapping);
    }

    /**
     * Indexes test documents with vector data for late interaction scoring tests.
     * Creates documents with different vector values to test scoring behavior.
     */
    private void indexDocuments() throws Exception {
        // Document 1: vectors [[0.3, 0.4]]
        List<List<Double>> docVectors1 = new ArrayList<>();
        List<Double> dv1 = new ArrayList<>();
        dv1.add(0.3);
        dv1.add(0.4);
        docVectors1.add(dv1);
        addKnnDoc(INDEX_NAME, "1", "my_vector", docVectors1);

        // Document 2: vectors [[0.1, 0.2]]
        List<List<Double>> docVectors2 = new ArrayList<>();
        List<Double> dv2 = new ArrayList<>();
        dv2.add(0.1);
        dv2.add(0.2);
        docVectors2.add(dv2);
        addKnnDoc(INDEX_NAME, "2", "my_vector", docVectors2);

        refreshAllIndices();
    }
}
