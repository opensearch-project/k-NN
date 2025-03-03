/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class KNNMapperSearcherIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "test_index";
    private static final String FIELD_NAME = "test_vector";

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

    /**
     * Request:
     * {
     *   "stored_fields": ["test_vector"],
     *   "query": {
     *     "match_all": {}
     *   }
     * }
     *
     * Example Response:
     * {
     *   "took":248,
     *   "timed_out":false,
     *   "_shards":{
     *     "total":1,
     *     "successful":1,
     *     "skipped":0,
     *     "failed":0
     *   },
     *   "hits":{
     *     "total":{
     *       "value":1,
     *       "relation":"eq"
     *     },
     *     "max_score":1.0,
     *     "hits":[
     *       {
     *         "_index":"test_index",
     *         "_id":"1",
     *         "_score":1.0,
     *         "fields":{"test_vector":[[-128,0,1,127]]}
     *       }
     *     ]
     *   }
     * }
     */
    @SneakyThrows
    public void testStoredFields_whenByteDataType_thenSucceed() {
        // Create index with stored field and confirm that we can properly retrieve it
        int[] testVector = new int[] { -128, 0, 1, 127 };
        String expectedResponse = String.format("\"fields\":{\"%s\":[[-128,0,1,127]]}}", FIELD_NAME);
        createKnnIndex(
            INDEX_NAME,
            createVectorMapping(testVector.length, KNNEngine.LUCENE.getName(), VectorDataType.BYTE.getValue(), true)
        );
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, testVector);

        final XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field(STORED_QUERY_FIELD, List.of(FIELD_NAME));
        builder.startObject(QUERY);
        builder.startObject(MATCH_ALL_QUERY_FIELD);
        builder.endObject();
        builder.endObject();
        builder.endObject();

        String response = EntityUtils.toString(performSearch(INDEX_NAME, builder.toString()).getEntity());
        assertTrue(response.contains(expectedResponse));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Request:
     * {
     *   "stored_fields": ["test_vector"],
     *   "query": {
     *     "match_all": {}
     *   }
     * }
     *
     * Example Response:
     * {
     *   "took":248,
     *   "timed_out":false,
     *   "_shards":{
     *     "total":1,
     *     "successful":1,
     *     "skipped":0,
     *     "failed":0
     *   },
     *   "hits":{
     *     "total":{
     *       "value":1,
     *       "relation":"eq"
     *     },
     *     "max_score":1.0,
     *     "hits":[
     *       {
     *         "_index":"test_index",
     *         "_id":"1",
     *         "_score":1.0,
     *         "fields":{"test_vector":[[-100.0,100.0,0.0,1.0]]}
     *       }
     *     ]
     *   }
     * }
     */
    @SneakyThrows
    public void testStoredFields_whenFloatDataType_thenSucceed() {
        List<KNNEngine> enginesToTest = List.of(KNNEngine.FAISS, KNNEngine.LUCENE);
        float[] testVector = new float[] { -100.0f, 100.0f, 0f, 1f };
        String expectedResponse = String.format("\"fields\":{\"%s\":[[-100.0,100.0,0.0,1.0]]}}", FIELD_NAME);
        for (KNNEngine knnEngine : enginesToTest) {
            createKnnIndex(INDEX_NAME, createVectorMapping(testVector.length, knnEngine.getName(), VectorDataType.FLOAT.getValue(), true));
            addKnnDoc(INDEX_NAME, "1", FIELD_NAME, testVector);

            final XContentBuilder builder = XContentFactory.jsonBuilder();
            builder.startObject();
            builder.field(STORED_QUERY_FIELD, List.of(FIELD_NAME));
            builder.startObject(QUERY);
            builder.startObject(MATCH_ALL_QUERY_FIELD);
            builder.endObject();
            builder.endObject();
            builder.endObject();

            String response = EntityUtils.toString(performSearch(INDEX_NAME, builder.toString()).getEntity());
            assertTrue(response.contains(expectedResponse));

            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testPutMappings_whenIndexAlreadyCreated_thenSuccess() {
        List<KNNEngine> enginesToTest = List.of(KNNEngine.FAISS, KNNEngine.LUCENE);
        float[] testVector = new float[] { -100.0f, 100.0f, 0f, 1f };
        for (KNNEngine knnEngine : enginesToTest) {
            String indexName = INDEX_NAME + "_" + knnEngine.getName();
            createKnnIndex(indexName, createVectorMapping(testVector.length, knnEngine.getName(), VectorDataType.FLOAT.getValue(), false));
            putMappingRequest(
                indexName,
                createVectorMapping(testVector.length, knnEngine.getName(), VectorDataType.FLOAT.getValue(), false)
            );
        }
        // Check with FlatMapper
        String indexName = INDEX_NAME + "_flat_index";
        createBasicKnnIndex(indexName, FIELD_NAME, testVector.length);
        putMappingRequest(indexName, createKnnIndexMapping(FIELD_NAME, testVector.length));
    }

    /**
     * Mapping
     * {
     *     "properties": {
     *         "test_vector": {
     *             "type": "knn_vector",
     *             "dimension": {dimension},
     *             "data_type": "{type}",
     *             "stored": true
     *             "method": {
     *                 "name": "hnsw",
     *                 "engine": "{engine}"
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    private String createVectorMapping(final int dimension, final String engine, final String dataType, final boolean isStored) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, dataType)
            .field(STORE_FIELD, isStored)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }

}
