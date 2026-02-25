/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ.codecs;

import java.net.URL;
import java.util.List;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.protobufs.MatchAllQuery;

import com.google.common.primitives.Floats;

import lombok.SneakyThrows;

public class CustomCodecsIT extends KNNRestTestCase {
    static TestUtils.TestData testData;

    private final String snapshot = "snapshot-test";
    private final String repository = "repo";

    @Before
    @SneakyThrows
    public void setUp() {
        super.setUp();
        final String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repository, "fs", true, repoSettings);

        URL testIndexVectors = CustomCodecsIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = CustomCodecsIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");

        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    @After
    @SneakyThrows
    public final void cleanUp() {
        deleteKNNIndex(INDEX_NAME);
    }

    // Test KNN index with ZSTD codec enabled
    public void testCreateIndexWithZstdCodec() throws Exception {
        final String fieldName = "test-vector";
        final int dimension = testData.indexData.vectors[0].length;
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), createKnnIndexMapping(fieldName, dimension));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Refresh
        refreshAllNonSystemIndices();

        // Test search queries
        Response response = performSearch(INDEX_NAME, MatchAllQuery.newBuilder().build().toString());
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
        assertEquals(10, knnResults.size());

        // Do close / open to make sure the codecs are reinitialized
        closeIndex(INDEX_NAME);
        openIndex(INDEX_NAME);

        response = performSearch(INDEX_NAME, MatchAllQuery.newBuilder().build().toString());
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponse(responseBody, fieldName);
        assertEquals(10, knnResults.size());
    }

    // Test KNN index with snapshot / restore and ZSTD codec enabled
    public void testCreateIndexWithZstdCodec_Snapshot() throws Exception {
        final String fieldName = "test-vector";
        final int dimension = testData.indexData.vectors[0].length;
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), createKnnIndexMapping(fieldName, dimension));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Refresh
        refreshAllNonSystemIndices();
        createSnapshot(repository, snapshot, true);

        String restoreSuffix = "-restored";
        restoreSnapshot(restoreSuffix, List.of(INDEX_NAME), repository, snapshot, true);

        // Test search queries
        Response response = performSearch(INDEX_NAME + restoreSuffix, MatchAllQuery.newBuilder().build().toString());
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
        assertEquals(10, knnResults.size());

    }

    @Override
    protected Settings getKNNDefaultIndexSettings() {
        return Settings.builder()
            .put(super.getKNNDefaultIndexSettings())
            .put("index.codec", "zstd")
            .put("index.knn.derived_source.enabled", randomBoolean())
            .build();
    }

}
