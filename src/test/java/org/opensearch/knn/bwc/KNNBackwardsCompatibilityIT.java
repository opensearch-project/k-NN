/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.*;
import java.util.stream.Collectors;
import org.apache.http.util.EntityUtils;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.junit.Assert;
import org.opensearch.common.settings.Settings;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.lang.*;

import static org.opensearch.knn.TestUtils.*;

public class KNNBackwardsCompatibilityIT extends KNNRestTestCase {
    private static final String CLUSTER_NAME = System.getProperty(TEST_CLUSTER_NAME);
    private final String testIndexName = KNN_BWC_PREFIX+"test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    @Override
    protected final boolean preserveIndicesUponCompletion() {
        return true;
    }

    @Override
    protected final boolean preserveReposUponCompletion() {
        return true;
    }

    @Override
    protected boolean preserveTemplatesUponCompletion() {
        return true;
    }

    @Override
    protected final Settings restClientSettings() {
        return Settings
                .builder()
                .put(super.restClientSettings())
                // increase the timeout here to 90 seconds to handle long waits for a green
                // cluster health. the waits for green need to be longer than a minute to
                // account for delayed shards
                .put(OpenSearchRestTestCase.CLIENT_SOCKET_TIMEOUT, "90s")
                .build();
    }

    private enum ClusterType {
        OLD,
        MIXED,
        UPGRADED;

        public static ClusterType parse(String value) {
            switch (value) {
                case "old_cluster":
                    return OLD;
                case "mixed_cluster":
                    return MIXED;
                case "upgraded_cluster":
                    return UPGRADED;
                default:
                    throw new IllegalArgumentException("unknown cluster type: " + value);
            }
        }
    }

    private ClusterType getClusterType(){
        return ClusterType.parse(System.getProperty(BWCSUITE_CLUSTER));
    }

// Use this prefix "knn-bwc-" while creating a test index to test BWC Tests.
// For example:  testIndexName = "knn-bwc-test-index"
    @SuppressWarnings("unchecked")
    public void testBackwardsCompatibility() throws Exception {
        String uri = getUri(getClusterType());
        Map<String, Map<String, Object>> responseMap = (Map<String, Map<String, Object>>) getAsMap(uri).get("nodes");
        for (Map<String, Object> response : responseMap.values()) {
            List<Map<String, Object>> plugins = (List<Map<String, Object>>) response.get("plugins");
            Set<Object> pluginNames = plugins.stream().map(map -> map.get("name")).collect(Collectors.toSet());
            switch (getClusterType()) {
                case OLD:
                    Assert.assertTrue(pluginNames.contains(OS_KNN));

                    Request waitForGreen = new Request("GET", "/_cluster/health");
                    waitForGreen.addParameter("wait_for_nodes", "3");
                    waitForGreen.addParameter("wait_for_status", "green");
                    client().performRequest(waitForGreen);

                    int graphCountBefore = getTotalGraphsInCache();

                    createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));
                    addKnnDoc(testIndexName, "1", testFieldName, new Float[]{6.0f, 6.0f});

                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCountBefore + 1, getTotalGraphsInCache());

                    break;
                case MIXED:
                    Assert.assertTrue(pluginNames.contains(OS_KNN));

                    float[] queryVector = {10.0f, 10.0f};
                    int k = 1;

                    KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(testFieldName, queryVector, k);
                    Response resp = searchKNNIndex(testIndexName, knnQueryBuilder,k);
                    List<KNNResult> results = parseSearchResponse(EntityUtils.toString(resp.getEntity()), testFieldName);

                    assertEquals(results.size(), k);
                    String round = System.getProperty(BWCSUITE_ROUND);
                    if(round.equals("first")) {
                        assertEquals("1", results.get(0).getDocId());
                        int graphCountFirst = getTotalGraphsInCache();

                        deleteKnnDoc(testIndexName, "1");
                        addKnnDoc(testIndexName, "2", testFieldName, new Float[]{50.0f, 50.0f});
                        addKnnDoc(testIndexName, "4", testFieldName, new Float[]{55.0f, 55.0f});

                        knnWarmup(Collections.singletonList(testIndexName));
                        assertEquals(graphCountFirst + 2, getTotalGraphsInCache());
                    }
                    else if(round.equals("second")) {
                        assertEquals("2", results.get(0).getDocId());
                        int graphCountSecond = getTotalGraphsInCache();

                        deleteKnnDoc(testIndexName, "2");

                        knnWarmup(Collections.singletonList(testIndexName));
                        assertEquals(graphCountSecond, getTotalGraphsInCache());
                    }
                    else {
                        assertEquals("4", results.get(0).getDocId());
                        deleteKNNIndex(testIndexName);
                    }

                    break;
                case UPGRADED:
                    Assert.assertTrue(pluginNames.contains(OS_KNN));

                    int graphCountUpgraded = getTotalGraphsInCache();

                    updateKnnDoc(testIndexName, "1", testFieldName, new Float[]{17.0f, 17.0f});

                    addKnnDoc(testIndexName, "3", testFieldName, new Float[]{20.0f, 20.0f});

                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCountUpgraded+3, getTotalGraphsInCache());

                    forceMergeKnnIndex(testIndexName);

                    float[] queryVector1 = {15.0f, 15.0f};
                    int k1 = 1;

                    KNNQueryBuilder knnQueryBuilder1 = new KNNQueryBuilder(testFieldName, queryVector1, k1);
                    Response resp1 = searchKNNIndex(testIndexName, knnQueryBuilder1,k1);
                    List<KNNResult> results1 = parseSearchResponse(EntityUtils.toString(resp1.getEntity()), testFieldName);

                    assertEquals(results1.size(), k1);
                    assertEquals("1", results1.get(0).getDocId());

                    deleteKNNIndex(testIndexName);

                    break;
            }
            break;
        }
    }

    private String getUri(ClusterType clusterType) {
        switch (clusterType) {
            case OLD:
                return String.join("/","_nodes", CLUSTER_NAME + "-0", "plugins");
            case MIXED:
                String round = System.getProperty(BWCSUITE_ROUND);
                if (round.equals("second")) {
                    return String.join("/","_nodes", CLUSTER_NAME + "-1", "plugins");
                } if (round.equals("third")) {
                    return String.join("/","_nodes", CLUSTER_NAME + "-2", "plugins");
                }
                    return String.join("/","_nodes", CLUSTER_NAME + "-0", "plugins");

            case UPGRADED:
                return "_nodes/plugins";
            default:
                throw new IllegalArgumentException("unknown cluster type: " + clusterType);
        }
    }
}
