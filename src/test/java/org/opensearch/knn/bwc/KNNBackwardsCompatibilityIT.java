/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.ArrayList;
import org.apache.http.util.EntityUtils;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.KNNQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.junit.Assert;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.test.rest.OpenSearchRestTestCase;
import static org.opensearch.knn.TestUtils.*;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;

public class KNNBackwardsCompatibilityIT extends KNNRestTestCase {
    private static final String CLUSTER_NAME = System.getProperty(TEST_CLUSTER_NAME);
    private final String testIndexName = KNN_BWC_PREFIX + "test-index";
    private final String testIndex_Recall = KNN_BWC_PREFIX + "test-index-recall";
    private final String testIndex_NullParams = KNN_BWC_PREFIX + "test-index-null-params";
    private final String testFieldName = "test-field";
    private final String testField_Recall = "test-field-recall";
    private final String testIndex_Recall_Old = KNN_BWC_PREFIX + "test-index-recall-value-old";
    private final int dimensions_Recall_Old = 1;
    private final int dimensions = 2;
    private final int dimensions_Recall = 50;
    private final int docCount = 1000;
    private final int queryCount = 100;
    private final int k_Recall = 10;

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

                    addDocs(testIndex_Recall, testField_Recall, dimensions_Recall, docCount, true);
                    double recallVal = getkNNBWCRecallValue(testIndex_Recall, testField_Recall, docCount, dimensions_Recall, queryCount, k_Recall, true, SpaceType.L2);
                    createKnnIndex(testIndex_Recall_Old, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions_Recall_Old));
                    addKnnDoc(testIndex_Recall_Old, "1", testFieldName, new Float[]{(float) recallVal});

                    break;
                case MIXED:
                    Assert.assertTrue(pluginNames.contains(OS_KNN));

                    float[] queryVector = {10.0f, 10.0f};
                    int k = 1;

                    KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(testFieldName, queryVector, k);
                    Response resp = searchKNNIndex(testIndexName, knnQueryBuilder,k);
                    List<KNNResult> results = parseSearchResponse(EntityUtils.toString(resp.getEntity()), testFieldName);

                    assertEquals(k, results.size());
                    String round = System.getProperty(BWCSUITE_ROUND);
                    if ("first".equals(round)) {
                        assertEquals("1", results.get(0).getDocId());
                        int graphCountFirst = getTotalGraphsInCache();

                        deleteKnnDoc(testIndexName, "1");
                        addKnnDoc(testIndexName, "2", testFieldName, new Float[]{50.0f, 50.0f});
                        addKnnDoc(testIndexName, "4", testFieldName, new Float[]{55.0f, 55.0f});

                        knnWarmup(Collections.singletonList(testIndexName));
                        assertEquals(graphCountFirst + 2, getTotalGraphsInCache());
                    } else if ("second".equals(round)) {
                        assertEquals("2", results.get(0).getDocId());
                        int graphCountSecond = getTotalGraphsInCache();

                        deleteKnnDoc(testIndexName, "2");

                        knnWarmup(Collections.singletonList(testIndexName));
                        assertEquals(graphCountSecond, getTotalGraphsInCache());
                    } else {
                        assertEquals("4", results.get(0).getDocId());
                    }

                    double recallValMixed = getkNNBWCRecallValue(testIndex_Recall, testField_Recall, docCount, dimensions_Recall, queryCount, k_Recall, true, SpaceType.L2);
                    float[][] expRecallVal = getIndexVectorsFromIndex(testIndex_Recall_Old, testFieldName, 1, dimensions_Recall_Old);
                    assertEquals(expRecallVal[0][0], recallValMixed, 0.2);

                    if ("third".equals(round)) {
                        deleteKNNIndex(testIndexName);
                        deleteKNNIndex(testIndex_Recall);
                        deleteKNNIndex(testIndex_Recall_Old);
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

                    double recallValUpgraded = getkNNBWCRecallValue(testIndex_Recall, testField_Recall, docCount, dimensions_Recall, queryCount, k_Recall, true, SpaceType.L2);
                    float[][] expRecallValue = getIndexVectorsFromIndex(testIndex_Recall_Old, testFieldName, 1, dimensions_Recall_Old);
                    assertEquals(expRecallValue[0][0], recallValUpgraded, 0.2);

                    deleteKNNIndex(testIndexName);
                    deleteKNNIndex(testIndex_Recall);
                    deleteKNNIndex(testIndex_Recall_Old);

                    break;
            }
            break;
        }
    }

    public void testNullParametersOnUpgrade() throws Exception {

        // Skip test if version is 1.2 or 1.3
        // systemProperty 'tests.plugin_bwc_version', knn_bwc_version
        String bwcVersion = System.getProperty("tests.plugin_bwc_version", null);
        if (bwcVersion == null || bwcVersion.startsWith("1.2") || bwcVersion.startsWith("1.3")) {
            return;
        }

        // Confirm cluster is green before starting
        Request waitForGreen = new Request("GET", "/_cluster/health");
        waitForGreen.addParameter("wait_for_nodes", "3");
        waitForGreen.addParameter("wait_for_status", "green");
        client().performRequest(waitForGreen);

        switch (getClusterType()) {
            case OLD:
                String mapping = Strings.toString(
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("properties")
                        .startObject(testFieldName)
                        .field("type", "knn_vector")
                        .field("dimension", String.valueOf(dimensions))
                        .startObject(KNN_METHOD)
                        .field(NAME, METHOD_HNSW)
                        .field(PARAMETERS, (String) null)
                        .endObject()
                        .endObject()
                        .endObject()
                        .endObject()
                );

                createKnnIndex(testIndex_NullParams, getKNNDefaultIndexSettings(), mapping);
                break;
            case UPGRADED:
                deleteKNNIndex(testIndex_NullParams);
                break;
        }
    }

    private String getUri(ClusterType clusterType) {
        switch (clusterType) {
            case OLD:
                return String.join("/","_nodes", CLUSTER_NAME + "-0", "plugins");
            case MIXED:
                String round = System.getProperty(BWCSUITE_ROUND);
                if ("second".equals(round)) {
                    return String.join("/","_nodes", CLUSTER_NAME + "-1", "plugins");
                }
                if ("third".equals(round)) {
                    return String.join("/","_nodes", CLUSTER_NAME + "-2", "plugins");
                }
                return String.join("/","_nodes", CLUSTER_NAME + "-0", "plugins");

            case UPGRADED:
                return "_nodes/plugins";
            default:
                throw new IllegalArgumentException("unknown cluster type: " + clusterType);
        }
    }

   private double getkNNBWCRecallValue(String testIndex, String testField, int docCount, int dimensions, int queryCount, int k, boolean isStandard, SpaceType spaceType) throws Exception {
       float[][] indexVectors = getIndexVectorsFromIndex(testIndex, testField, docCount, dimensions);
       float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, docCount, isStandard);
       List<Set<String>> groundTruthValues = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, spaceType, k);
       List<List<String>> searchResults = bulkSearch(testIndex, testField, queryVectors, k);
       return TestUtils.calculateRecallValue(searchResults, groundTruthValues, k);
   }

    private void addDocs(String testIndex, String testField, int dimensions, int docCount, boolean isStandard) throws Exception {
        createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(testField, dimensions));

        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 2);
        updateClusterSettings(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true);

        bulkAddKnnDocs(testIndex, testField, TestUtils.getIndexVectors(docCount, dimensions, isStandard), docCount);
    }

}
