/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.action;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.junit.Test;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

/**
 * Integration tests to check the correctness of Legacy KNN Warmup API
 */

public class RestLegacyKNNWarmupHandlerIT extends KNNRestTestCase {

    public static final int ALWAYS_BUILD_GRAPH = 0;
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    @Test(expected = ResponseException.class)
    public void testNonExistentIndex() throws IOException {
        executeWarmupRequest(Collections.singletonList("non-existent"), KNNPlugin.LEGACY_KNN_BASE_URI);
    }

    @Test(expected = ResponseException.class)
    public void testNonKnnIndex() throws IOException {
        createIndex("not-knn-index", Settings.EMPTY);

        executeWarmupRequest(Collections.singletonList("not-knn-index"), KNNPlugin.LEGACY_KNN_BASE_URI);
    }

    public void testEmptyIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(testFieldName, dimensions));

        executeWarmupRequest(Collections.singletonList(testIndexName), KNNPlugin.LEGACY_KNN_BASE_URI);

        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    public void testSingleIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        executeWarmupRequest(Collections.singletonList(testIndexName), KNNPlugin.LEGACY_KNN_BASE_URI);

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
    }

    public void testMultipleIndices() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName + "1", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "1", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        createKnnIndex(testIndexName + "2", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "2", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        executeWarmupRequest(Arrays.asList(testIndexName + "1", testIndexName + "2"), KNNPlugin.LEGACY_KNN_BASE_URI);

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());
    }
}
