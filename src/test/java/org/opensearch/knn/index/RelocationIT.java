/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.junit.BeforeClass;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.common.settings.Settings;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

public class RelocationIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "relocation-knn-index";
    private static final String TEST_FIELD = "test-field";

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (FaissIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of RelocationIT Class is null");
        }
    }

    public void testForcemerge_whenRelocation_and_abortSuccess() throws Exception {

        int dimension = 768;
        int numNodes = 2;
        Request nodesNamesRequest = new Request("GET", "_cat/nodes?h=name");
        Response response = client().performRequest(nodesNamesRequest);
        assertOK(response);
        String[] nodeNamesStr = new String(response.getEntity().getContent().readAllBytes()).split("\n");
        if (nodeNamesStr.length < numNodes) {
            return;
        }
        Settings indexSettings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.routing.allocation.require._name", nodeNamesStr[0])
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 10000)
            .put(KNN_INDEX, true)
            .build();

        createKnnIndex(
            INDEX_NAME,
            indexSettings,
            createKnnIndexMapping(
                TEST_FIELD,
                dimension,
                METHOD_HNSW,
                KNNEngine.FAISS.getName(),
                SpaceType.INNER_PRODUCT.getValue(),
                true,
                VectorDataType.FLOAT
            )
        );
        waitForClusterHealthGreen(Integer.toString(numNodes));

        logger.info("bulk Write docs");
        for (int i = 0; i < 10; i++) {
            addKNNDocsWithParkingAndRating(INDEX_NAME, TEST_FIELD, dimension, i * 1000, 1000);
            refreshAllIndices();
        }

        Thread write = new Thread(() -> {
            try {
                Settings.Builder updateSettings = Settings.builder().put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0);
                updateIndexSettings(INDEX_NAME, updateSettings);

                logger.info("Force merge");
                forceMergeKnnIndex(INDEX_NAME);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        write.start();

        Thread.sleep(1000 * 20);

        logger.info("Update Routing Allocation");
        Settings.Builder updateSettings = Settings.builder().put("index.routing.allocation.require._name", nodeNamesStr[1]);
        updateIndexSettings(INDEX_NAME, updateSettings);

        while (true) {
            Request request = new Request("GET", "_cat/recovery?active_only=true");

            Response resp = client().performRequest(request);
            String respStr = new String(resp.getEntity().getContent().readAllBytes());
            if (respStr.isEmpty()) {
                break;
            } else {
                logger.info("recovery: {}", respStr);
            }
            Thread.sleep(1000 * 1);
        }
        write.join();
    }
}
