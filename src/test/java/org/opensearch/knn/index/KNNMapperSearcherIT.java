/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.client.Response;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KNNMapperSearcherIT extends KNNRestTestCase {
    private static final Logger logger = LogManager.getLogger(KNNMapperSearcherIT.class);

    /**
     * Test Data set
     */
    private void addTestData() throws Exception {
        Float[] f1 = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, f1);

        Float[] f2 = { 2.0f, 2.0f };
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, f2);

        Float[] f3 = { 4.0f, 4.0f };
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, f3);

        Float[] f4 = { 3.0f, 3.0f };
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, f4);
    }

    public void testKNNResultsWithForceMerge() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();
        forceMergeKnnIndex(INDEX_NAME);

        /**
         * Query params
         */
        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);

        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(k, results.size());
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId());
        }
    }

    public void testKNNResultsUpdateDocAndForceMerge() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addDocWithNumericField(INDEX_NAME, "1", "abc", 100);
        addTestData();
        forceMergeKnnIndex(INDEX_NAME);

        /**
         * Query params
         */
        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);

        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(k, results.size());
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId());
        }
    }

    public void testKNNResultsWithoutForceMerge() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();

        /**
         * Query params
         */
        float[] queryVector = { 2.0f, 2.0f }; // vector to be queried
        int k = 3; // nearest 3 neighbors
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);

        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        List<String> expectedDocids = Arrays.asList("2", "4", "3");

        List<String> actualDocids = new ArrayList<>();
        for (KNNResult result : results) {
            actualDocids.add(result.getDocId());
        }

        assertEquals(actualDocids.size(), k);
        assertArrayEquals(actualDocids.toArray(), expectedDocids.toArray());
    }

    public void testKNNResultsWithNewDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId()); // Vector of DocId 2 is closest to the query
        }

        /**
         * Add new doc with vector not nearest than doc 2
         */
        Float[] newVector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "6", FIELD_NAME, newVector);
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId());
        }

        /**
         * Add new doc with vector nearest than doc 2 to queryVector
         */
        Float[] newVector1 = { 0.5f, 0.5f };
        addKnnDoc(INDEX_NAME, "7", FIELD_NAME, newVector1);
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("7", result.getDocId());
        }
    }

    public void testKNNResultsWithUpdateDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId()); // Vector of DocId 2 is closest to the query
        }

        /**
         * update doc 3 to the nearest
         */
        Float[] updatedVector = { 0.1f, 0.1f };
        updateKnnDoc(INDEX_NAME, "3", FIELD_NAME, updatedVector);
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("3", result.getDocId()); // Vector of DocId 3 is closest to the query
        }
    }

    public void testKNNResultsWithDeleteDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("2", result.getDocId()); // Vector of DocId 2 is closest to the query
        }

        /**
         * delete the nearest doc (doc2)
         */
        deleteKnnDoc(INDEX_NAME, "2");

        knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k + 1);
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(results.size(), k);
        for (KNNResult result : results) {
            assertEquals("4", result.getDocId()); // Vector of DocId 4 is closest to the query
        }
    }

    /**
     * For negative K, query builder should throw Exception
     */
    public void testNegativeK() {
        float[] vector = { 1.0f, 2.0f };
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, vector, -1));
    }

    /**
     *  For zero K, query builder should throw Exception
     */
    public void testZeroK() {
        float[] vector = { 1.0f, 2.0f };
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, vector, 0));
    }

    /**
     * K &gt; &gt; number of docs
     */
    public void testLargeK() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        addTestData();

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = KNNQueryBuilder.K_MAX; // nearest 1 neighbor

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(results.size(), 4);
    }
}
