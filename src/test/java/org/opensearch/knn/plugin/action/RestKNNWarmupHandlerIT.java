/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.action;

import org.opensearch.knn.KNNRestTestCase;
import org.junit.Test;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

/**
 * Integration tests to check the correctness of KNN Warmup API
 */

public class RestKNNWarmupHandlerIT extends KNNRestTestCase {

    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    @Test(expected = ResponseException.class)
    public void testNonExistentIndex() throws IOException {
        knnWarmup(Collections.singletonList("non-existent"));
    }

    @Test(expected = ResponseException.class)
    public void testNonKnnIndex() throws IOException {
        createIndex("not-knn-index", Settings.EMPTY);

        knnWarmup(Collections.singletonList("not-knn-index"));
    }

    public void testEmptyIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));

        knnWarmup(Collections.singletonList(testIndexName));

        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    public void testSingleIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        knnWarmup(Collections.singletonList(testIndexName));

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
    }

    public void testMultipleIndices() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName + "1", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "1", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        createKnnIndex(testIndexName + "2", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "2", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        knnWarmup(Arrays.asList(testIndexName + "1", testIndexName + "2"));

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());
    }

    public void testWarmIndex_skipsWarmup() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        // Create a k-NN index with index.warm=true to simulate warm-tier index
        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName, warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Warmup should succeed (HTTP 200) but skip loading graphs for the warm index
        knnWarmup(Collections.singletonList(testIndexName));

        // Graph count should not increase because warmup was skipped for warm-tier index
        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    public void testMixedWarmAndHotIndices_onlyWarmsHotIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        // Create a regular (hot) k-NN index
        createKnnIndex(testIndexName + "-hot", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-hot", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Create a warm-tier k-NN index
        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName + "-warm", warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-warm", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Warmup both indices — only the hot index should have its graphs loaded
        knnWarmup(Arrays.asList(testIndexName + "-hot", testIndexName + "-warm"));

        // Only 1 graph from the hot index should be added; warm index should be skipped
        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
    }
}
