/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.script.Script;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Verifies that the late_interaction_score_requests counter surfaced by the k-NN
 * _stats API increments ONCE PER QUERY (per shard), not once per scored document.
 *
 * Counting is done by {@code LateInteractionStatsSearchListener} in the query
 * phase, so a single query that scores many documents still counts as 1.
 */
public class LateInteractionScoreStatsIT extends KNNRestTestCase {

    private static final String STAT_NAME = "late_interaction_score_requests";

    /**
     * Verifies that the counter tracks adoption at the query level, not the document level.
     *
     * <p>The index is created with a single primary shard so the per-shard counter is
     * deterministic, and two multi-vector documents are indexed. Two late interaction
     * queries are then run over a match_all body that matches (and therefore scores) both
     * documents: the first with size=2 and the second with size=1. If counting were
     * per-document each query would add +2; instead each query must add exactly +1,
     * independent of how many documents are scored or returned. The stat progression is
     * therefore before -> before+1 -> before+2.
     */
    public void testLateInteractionScore_incrementsOncePerQueryNotPerDoc() throws Exception {
        createSingleShardIndex();
        indexDocuments();

        long before = getLateInteractionCount();

        runLateInteractionQuery(2);
        long afterFirst = getLateInteractionCount();
        assertEquals(
            "A single query must count once regardless of docs scored (before=" + before + ", after=" + afterFirst + ")",
            before + 1,
            afterFirst
        );

        runLateInteractionQuery(1);
        long afterSecond = getLateInteractionCount();
        assertEquals("Second query adds exactly +1 more (per-query, not per-doc)", before + 2, afterSecond);

        logger.info("{} progression: {} -> {} -> {}", STAT_NAME, before, afterFirst, afterSecond);
        deleteKNNIndex(INDEX_NAME);
    }

    private void runLateInteractionQuery(int size) throws Exception {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv1 = new ArrayList<>();
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
            size,
            Collections.emptyMap()
        );
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    @SuppressWarnings("unchecked")
    private long getLateInteractionCount() throws Exception {
        Response response = client().performRequest(new Request("GET", "/_plugins/_knn/stats"));
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        Map<String, Object> body = entityAsMap(response);
        Map<String, Object> nodes = (Map<String, Object>) body.get("nodes");
        long total = 0;
        if (nodes != null) {
            for (Object nodeObj : nodes.values()) {
                Map<String, Object> node = (Map<String, Object>) nodeObj;
                Object v = node.get(STAT_NAME);
                if (v instanceof Number) {
                    total += ((Number) v).longValue();
                }
            }
        }
        return total;
    }

    private void createSingleShardIndex() throws Exception {
        String body = "{\n"
            + "  \"settings\": { \"index\": { \"number_of_shards\": 1, \"number_of_replicas\": 0 } },\n"
            + "  \"mappings\": {\n"
            + "    \"properties\": {\n"
            + "      \"my_vector\": { \"type\": \"object\", \"enabled\": false }\n"
            + "    }\n"
            + "  }\n"
            + "}";
        Request request = new Request("PUT", "/" + INDEX_NAME);
        request.setJsonEntity(body);
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private void indexDocuments() throws Exception {
        List<List<Number>> docVectors1 = new ArrayList<>();
        List<Number> dv1 = new ArrayList<>();
        dv1.add(0.3);
        dv1.add(0.4);
        docVectors1.add(dv1);
        addKnnDoc(INDEX_NAME, "1", "my_vector", docVectors1);

        List<List<Number>> docVectors2 = new ArrayList<>();
        List<Number> dv2 = new ArrayList<>();
        dv2.add(0.1);
        dv2.add(0.2);
        docVectors2.add(dv2);
        addKnnDoc(INDEX_NAME, "2", "my_vector", docVectors2);

        refreshAllIndices();
    }
}
