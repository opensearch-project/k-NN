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
}
