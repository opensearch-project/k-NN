/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.action;

import lombok.SneakyThrows;
import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.rest.RestRequest;

import java.util.Arrays;
import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.CLEAR_CACHE;

/**
 * Integration tests to validate ClearCache API
 */

public class RestClearCacheHandlerIT extends KNNRestTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;
    public static final int ALWAYS_BUILD_GRAPH = 0;

    @SneakyThrows
    public void testNonExistentIndex() {
        String nonExistentIndex = "non-existent-index";

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, CLEAR_CACHE, nonExistentIndex);
        Request request = new Request(RestRequest.Method.POST.name(), restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains(nonExistentIndex));
    }

    @SneakyThrows
    public void testNotKnnIndex() {
        String notKNNIndex = "not-knn-index";
        createIndex(notKNNIndex, Settings.EMPTY);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, CLEAR_CACHE, notKNNIndex);
        Request request = new Request(RestRequest.Method.POST.name(), restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains(notKNNIndex));
    }

    @SneakyThrows
    public void testClearCacheSingleIndex() {
        String testIndex = getTestName().toLowerCase();
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndex, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        knnWarmup(Collections.singletonList(testIndex));

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());

        clearCache(Collections.singletonList(testIndex));
        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    @SneakyThrows
    public void testClearCacheMultipleIndices() {
        String testIndex1 = getTestName().toLowerCase();
        String testIndex2 = getTestName().toLowerCase() + 1;
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndex1, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex1, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        createKnnIndex(testIndex2, buildKNNIndexSettings(0), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex2, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        knnWarmup(Arrays.asList(testIndex1, testIndex2));

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());

        clearCache(Arrays.asList(testIndex1, testIndex2));
        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    @SneakyThrows
    public void testClearCacheMultipleIndicesWithPatterns() {
        String testIndex1 = getTestName().toLowerCase();
        String testIndex2 = getTestName().toLowerCase() + 1;
        String testIndex3 = "abc" + getTestName().toLowerCase();
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndex1, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex1, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        createKnnIndex(testIndex2, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex2, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        createKnnIndex(testIndex3, buildKNNIndexSettings(ALWAYS_BUILD_GRAPH), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex3, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        knnWarmup(Arrays.asList(testIndex1, testIndex2, testIndex3));

        assertEquals(graphCountBefore + 3, getTotalGraphsInCache());
        String indexPattern = getTestName().toLowerCase() + "*";

        clearCache(Arrays.asList(indexPattern));
        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
    }
}
