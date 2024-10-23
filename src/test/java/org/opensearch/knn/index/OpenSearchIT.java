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

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import com.google.common.primitives.Floats;
import java.util.Locale;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MAX;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MIN;

public class OpenSearchIT extends KNNRestTestCase {

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (OpenSearchIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of OpenSearchIT Class is null");
        }
        URL testIndexVectors = OpenSearchIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = OpenSearchIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    public void testEndToEnd() throws Exception {
        String indexName = "test-index-1";
        KNNEngine knnEngine1 = KNNEngine.NMSLIB;
        KNNEngine knnEngine2 = KNNEngine.FAISS;
        String fieldName1 = "test-field-1";
        String fieldName2 = "test-field-2";
        SpaceType spaceType1 = SpaceType.COSINESIMIL;
        SpaceType spaceType2 = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName1)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType1.getValue())
            .field(KNNConstants.KNN_ENGINE, knnEngine1.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .startObject(fieldName2)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType2.getValue())
            .field(KNNConstants.KNN_ENGINE, knnEngine2.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();
        createKnnIndex(indexName, buildKNNIndexSettings(0), mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName1, fieldName2),
                ImmutableList.of(
                    Floats.asList(testData.indexData.vectors[i]).toArray(),
                    Floats.asList(testData.indexData.vectors[i]).toArray()
                )
            );
        }

        // Assert we have the right number of documents in the index
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            // Search the first field
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName1, testData.queries[i], k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName1);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName1);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    knnEngine1.score(1 - KNNScoringUtil.cosinesimil(testData.queries[i], primitiveArray), spaceType1),
                    actualScores.get(j),
                    0.0001
                );
            }

            // Search the second field
            response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName2, testData.queries[i], k), k);
            responseBody = EntityUtils.toString(response.getEntity());
            knnResults = parseSearchResponse(responseBody, fieldName2);
            assertEquals(k, knnResults.size());

            actualScores = parseSearchResponseScore(responseBody, fieldName2);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    knnEngine2.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType2),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    public void testAddDoc_blockedWhenCbTrips() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        updateClusterSettings("knn.circuit_breaker.triggered", "true");

        Float[] vector = { 6.0f, 6.0f };
        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector));
        String expMessage =
            "Parsing the created knn vector fields prior to indexing has failed as the circuit breaker triggered.  This indicates that the cluster is low on memory resources and cannot index more documents at the moment. Check _plugins/_knn/stats for the circuit breaker status.";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));

        // reset
        updateClusterSettings("knn.circuit_breaker.triggered", "false");
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
    }

    public void testUpdateDoc_blockedWhenCbTrips() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // update
        updateClusterSettings("knn.circuit_breaker.triggered", "true");
        Float[] updatedVector = { 8.0f, 8.0f };
        ResponseException ex = expectThrows(ResponseException.class, () -> updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector));
        String expMessage =
            "Parsing the created knn vector fields prior to indexing has failed as the circuit breaker triggered.  This indicates that the cluster is low on memory resources and cannot index more documents at the moment. Check _plugins/_knn/stats for the circuit breaker status.";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));

        // reset
        updateClusterSettings("knn.circuit_breaker.triggered", "false");
        updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
    }

    public void testAddAndSearchIndex_whenCBTrips() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        for (int i = 1; i <= 4; i++) {
            Float[] vector = { (float) i, (float) (i + 1) };
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 10; // nearest 10 neighbor
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(4, results.size());

        updateClusterSettings("knn.circuit_breaker.triggered", "true");
        // Try add another doc
        Float[] vector = { 1.0f, 2.0f };
        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "5", FIELD_NAME, vector));

        // Still get 4 docs
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(4, results.size());

        updateClusterSettings("knn.circuit_breaker.triggered", "false");
        addKnnDoc(INDEX_NAME, "5", FIELD_NAME, vector);
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(5, results.size());
    }

    public void testIndexingVectorValidation_differentSizes() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 4));

        // valid case with 4 dimension
        Float[] vector = { 6.0f, 7.0f, 8.0f, 9.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // invalid case with lesser dimension than original (3 < 4)
        Float[] vector1 = { 6.0f, 7.0f, 8.0f };
        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector1));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Vector dimension mismatch. Expected: 4, Given: 3"));

        // invalid case with more dimension than original (5 > 4)
        Float[] vector2 = { 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector2));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Vector dimension mismatch. Expected: 4, Given: 5"));
    }

    @SneakyThrows
    public void testIndexingVectorValidation_zeroVector() {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();
        final boolean valid = randomBoolean();
        final String method = KNNConstants.METHOD_HNSW;
        String engine;
        String spaceType;
        if (valid) {
            engine = randomFrom(KNNEngine.values()).getName();
            spaceType = SpaceType.L2.getValue();
        } else {
            engine = randomFrom(KNNConstants.LUCENE_NAME, KNNConstants.NMSLIB_NAME);
            spaceType = SpaceType.COSINESIMIL.getValue();
        }
        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 4, method, engine, spaceType));
        Float[] zeroVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (valid) {
            addKnnDoc(INDEX_NAME, "1", FIELD_NAME, zeroVector);
        } else {
            ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "1", FIELD_NAME, zeroVector));
            assertTrue(
                EntityUtils.toString(ex.getResponse().getEntity())
                    .contains(
                        String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue())
                    )
            );
        }
    }

    public void testVectorMappingValidation_noDimension() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Exception ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, mapping));
        assertThat(ex.getMessage(), containsString("Dimension value missing for vector: " + FIELD_NAME));
    }

    public void testVectorMappingValidation_invalidDimension() {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        Exception ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndex(
                INDEX_NAME,
                settings,
                createKnnIndexMapping(FIELD_NAME, KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT) + 1)
            )
        );
        assertThat(
            ex.getMessage(),
            containsString(
                "Dimension value cannot be greater than "
                    + KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT)
                    + " for vector with engine: "
                    + KNNEngine.DEFAULT.getName()
            )
        );
    }

    public void testVectorMappingValidation_invalidVectorNaN() throws IOException {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { Float.NaN, Float.NaN };
        Exception ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector));
        assertThat(ex.getMessage(), containsString("KNN vector values cannot be NaN"));
    }

    public void testVectorMappingValidation_invalidVectorInfinity() throws IOException {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY };
        Exception ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector));
        assertThat(ex.getMessage(), containsString("KNN vector values cannot be infinity"));
    }

    public void testVectorMappingValidation_updateDimension() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 4));

        Exception ex = expectThrows(ResponseException.class, () -> putMappingRequest(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 5)));
        assertThat(ex.getMessage(), containsString("Cannot update parameter [dimension] from [4] to [5]"));
    }

    public void testVectorMappingValidation_multiFieldsDifferentDimension() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        String f4 = FIELD_NAME + "-4";
        String f5 = FIELD_NAME + "-5";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(f4)
            .field("type", "knn_vector")
            .field("dimension", "4")
            .endObject()
            .startObject(f5)
            .field("type", "knn_vector")
            .field("dimension", "5")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(INDEX_NAME, settings, mapping);

        // valid case with 4 dimension
        Float[] vector = { 6.0f, 7.0f, 8.0f, 9.0f };
        addKnnDoc(INDEX_NAME, "1", f4, vector);

        // valid case with 5 dimension
        Float[] vector1 = { 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        updateKnnDoc(INDEX_NAME, "1", f5, vector1);
    }

    public void testExistsQuery() throws Exception {
        String field1 = "field1";
        String field2 = "field2";
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(Arrays.asList(field1, field2), Arrays.asList(2, 2)));

        Float[] vector = { 6.0f, 7.0f };
        addKnnDoc(INDEX_NAME, "1", Arrays.asList(field1, field2), Arrays.asList(vector, vector));

        addKnnDoc(INDEX_NAME, "2", field1, vector);
        addKnnDoc(INDEX_NAME, "3", field1, vector);
        addKnnDoc(INDEX_NAME, "4", field1, vector);

        addKnnDoc(INDEX_NAME, "5", field2, vector);
        addKnnDoc(INDEX_NAME, "6", field2, vector);

        // Create document that does not have k-NN vector field
        Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/7?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("non-knn-field", "test").endObject();
        request.setJsonEntity(builder.toString());
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        ExistsQueryBuilder existsQueryBuilder = new ExistsQueryBuilder(field1);
        response = searchExists(INDEX_NAME, existsQueryBuilder, 10);

        assertEquals(4, parseTotalSearchHits(EntityUtils.toString(response.getEntity())));

        existsQueryBuilder = new ExistsQueryBuilder(field2);
        response = searchExists(INDEX_NAME, existsQueryBuilder, 10);

        assertEquals(3, parseTotalSearchHits(EntityUtils.toString(response.getEntity())));
    }

    public void testIndexingVectorValidation_updateVectorWithNull() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 4));

        // valid case with 4 dimension
        final Float[] vectorForDocumentOne = { 6.0f, 7.0f, 8.0f, 9.0f };
        final String docOneId = "1";
        addKnnDoc(INDEX_NAME, docOneId, FIELD_NAME, vectorForDocumentOne);

        final Float[] vectorForDocumentTwo = { 2.0f, 1.0f, 3.8f, 2.5f };
        final String docTwoId = "2";
        addKnnDoc(INDEX_NAME, docTwoId, FIELD_NAME, vectorForDocumentTwo);

        // checking that both documents are retrievable based on knn search query
        int k = 2;
        float[] queryVector = { 5.0f, 6.0f, 7.0f, 10.0f };
        final KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        final Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        final List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(2, results.size());
        assertEquals(docOneId, results.get(0).getDocId());
        assertEquals(docTwoId, results.get(1).getDocId());

        // update vector value to null
        updateKnnDoc(INDEX_NAME, docOneId, FIELD_NAME, null);

        // retrieving updated document by id, vector should be null
        final Map<String, Object> knnDocMapUpdated = getKnnDoc(INDEX_NAME, docOneId);
        assertNull(knnDocMapUpdated.get(FIELD_NAME));

        // checking that first document one is no longer returned by knn search
        final Response updatedResponse = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        final List<KNNResult> updatedResults = parseSearchResponse(EntityUtils.toString(updatedResponse.getEntity()), FIELD_NAME);
        assertEquals(1, updatedResults.size());
        assertEquals(docTwoId, updatedResults.get(0).getDocId());

        // update vector back to original value
        updateKnnDoc(INDEX_NAME, docOneId, FIELD_NAME, vectorForDocumentOne);
        final Response restoreInitialVectorValueResponse = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        final List<KNNResult> restoreInitialVectorValueResults = parseSearchResponse(
            EntityUtils.toString(restoreInitialVectorValueResponse.getEntity()),
            FIELD_NAME
        );
        assertEquals(2, restoreInitialVectorValueResults.size());
        assertEquals(docOneId, results.get(0).getDocId());
        assertEquals(docTwoId, results.get(1).getDocId());

        // retrieving updated document by id, vector should be not null but has the original value
        final Map<String, Object> knnDocMapRestoreInitialVectorValue = getKnnDoc(INDEX_NAME, docOneId);
        assertNotNull(knnDocMapRestoreInitialVectorValue.get(FIELD_NAME));
        final Float[] vectorRestoreInitialValue = ((List<Double>) knnDocMapRestoreInitialVectorValue.get(FIELD_NAME)).stream()
            .map(Double::floatValue)
            .toArray(Float[]::new);
        assertArrayEquals(vectorForDocumentOne, vectorRestoreInitialValue);
    }

    // This doesn't work since indices that are created post 2.17 don't evict by default when indices are closed or deleted.
    // Enable this PR once https://github.com/opensearch-project/k-NN/issues/2148 is resolved.
    @Ignore
    public void testCacheClear_whenCloseIndex() throws Exception {
        String indexName = "test-index-1";
        KNNEngine knnEngine1 = KNNEngine.NMSLIB;
        KNNEngine knnEngine2 = KNNEngine.FAISS;
        String fieldName1 = "test-field-1";
        String fieldName2 = "test-field-2";
        SpaceType spaceType1 = SpaceType.COSINESIMIL;
        SpaceType spaceType2 = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName1)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType1.getValue())
            .field(KNNConstants.KNN_ENGINE, knnEngine1.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .startObject(fieldName2)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType2.getValue())
            .field(KNNConstants.KNN_ENGINE, knnEngine2.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();
        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName1, fieldName2),
                ImmutableList.of(
                    Floats.asList(testData.indexData.vectors[i]).toArray(),
                    Floats.asList(testData.indexData.vectors[i]).toArray()
                )
            );
        }

        // Assert we have the right number of documents in the index
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            // Search the first field
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName1, testData.queries[i], k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName1);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName1);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    knnEngine1.score(1 - KNNScoringUtil.cosinesimil(testData.queries[i], primitiveArray), spaceType1),
                    actualScores.get(j),
                    0.0001
                );
            }

            // Search the second field
            response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName2, testData.queries[i], k), k);
            responseBody = EntityUtils.toString(response.getEntity());
            knnResults = parseSearchResponse(responseBody, fieldName2);
            assertEquals(k, knnResults.size());

            actualScores = parseSearchResponseScore(responseBody, fieldName2);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    knnEngine2.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType2),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Get Stats
        int graphCount = getTotalGraphsInCache();
        assertTrue(graphCount > 0);
        // Close index
        closeKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    public void testKNNIndex_whenBuildGraphThresholdIsPresent_thenGetThresholdValue() throws Exception {
        final Integer buildVectorDataStructureThreshold = randomIntBetween(
            INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MIN,
            INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MAX
        );
        final Settings settings = Settings.builder().put(buildKNNIndexSettings(buildVectorDataStructureThreshold)).build();
        final String knnIndexMapping = createKnnIndexMapping(FIELD_NAME, KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT));
        final String indexName = "test-index-with-build-graph-settings";
        createKnnIndex(indexName, settings, knnIndexMapping);
        final String buildVectorDataStructureThresholdSetting = getIndexSettingByName(
            indexName,
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD
        );
        assertNotNull("build_vector_data_structure_threshold index setting is not found", buildVectorDataStructureThresholdSetting);
        assertEquals(
            "incorrect setting for build_vector_data_structure_threshold",
            buildVectorDataStructureThreshold,
            Integer.valueOf(buildVectorDataStructureThresholdSetting)
        );
        deleteKNNIndex(indexName);
    }

    public void testKNNIndex_whenBuildThresholdIsNotProvided_thenShouldNotReturnSetting() throws Exception {
        final String knnIndexMapping = createKnnIndexMapping(FIELD_NAME, KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT));
        final String indexName = "test-index-with-build-graph-settings";
        createKnnIndex(indexName, knnIndexMapping);
        final String buildVectorDataStructureThresholdSetting = getIndexSettingByName(
            indexName,
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD
        );
        assertNull(
            "build_vector_data_structure_threshold index setting should not be added in index setting",
            buildVectorDataStructureThresholdSetting
        );
        deleteKNNIndex(indexName);
    }

    public void testKNNIndex_whenGetIndexSettingWithDefaultIsCalled_thenReturnDefaultBuildGraphThresholdValue() throws Exception {
        final String knnIndexMapping = createKnnIndexMapping(FIELD_NAME, KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT));
        final String indexName = "test-index-with-build-vector-graph-settings";
        createKnnIndex(indexName, knnIndexMapping);
        final String buildVectorDataStructureThresholdSetting = getIndexSettingByName(
            indexName,
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD,
            true
        );
        assertNotNull("build_vector_data_structure index setting is not found", buildVectorDataStructureThresholdSetting);
        assertEquals(
            "incorrect default setting for build_vector_data_structure_threshold",
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE,
            Integer.valueOf(buildVectorDataStructureThresholdSetting)
        );
        deleteKNNIndex(indexName);
    }

    /*
        For this testcase, we will create index with setting build_vector_data_structure_threshold as -1, then index few documents, perform knn search,
        then, confirm hits because of exact search though there are no graph. In next step, update setting to 0, force merge segment to 1, perform knn search and confirm expected
        hits are returned.
     */
    public void testKNNIndex_whenBuildVectorGraphThresholdIsProvidedEndToEnd_thenBuildGraphBasedOnSetting() throws Exception {
        final String indexName = "test-index-1";
        final String fieldName1 = "test-field-1";
        final String fieldName2 = "test-field-2";

        final Integer dimension = testData.indexData.vectors[0].length;
        final Settings knnIndexSettings = buildKNNIndexSettings(-1);

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName1)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .startObject(fieldName2)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(KNNConstants.PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, knnIndexSettings, builder.toString());

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName1, fieldName2),
                ImmutableList.of(
                    Floats.asList(testData.indexData.vectors[i]).toArray(),
                    Floats.asList(testData.indexData.vectors[i]).toArray()
                )
            );
        }

        refreshAllIndices();
        // Assert we have the right number of documents in the index
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        final List<KNNResult> nmslibNeighbors = getResults(indexName, fieldName1, testData.queries[0], 1);
        assertEquals("unexpected neighbors are returned", nmslibNeighbors.size(), nmslibNeighbors.size());

        final List<KNNResult> faissNeighbors = getResults(indexName, fieldName2, testData.queries[0], 1);
        assertEquals("unexpected neighbors are returned", faissNeighbors.size(), faissNeighbors.size());

        // update build vector data structure setting
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0));
        forceMergeKnnIndex(indexName, 1);

        final int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            // Search nmslib field
            final Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName1, testData.queries[i], k), k);
            final String responseBody = EntityUtils.toString(response.getEntity());
            final List<KNNResult> nmslibValidNeighbors = parseSearchResponse(responseBody, fieldName1);
            assertEquals(k, nmslibValidNeighbors.size());
            // Search faiss field
            final List<KNNResult> faissValidNeighbors = getResults(indexName, fieldName2, testData.queries[i], k);
            assertEquals(k, faissValidNeighbors.size());
        }

        // Delete index
        deleteKNNIndex(indexName);
    }

    /*
        For this testcase, we will create index with setting build_vector_data_structure_threshold number of documents to ingest, then index x documents, perform knn search,
        then, confirm expected hits are returned. Here, we don't need force merge to build graph, since, threshold is less than
        actual number of documents in segments
     */
    public void testKNNIndex_whenBuildVectorDataStructureIsLessThanDocCount_thenBuildGraphBasedSuccessfully() throws Exception {
        final String indexName = "test-index-1";
        final String fieldName = "test-field-1";

        final Integer dimension = testData.indexData.vectors[0].length;
        final Settings knnIndexSettings = buildKNNIndexSettings(testData.indexData.docs.length);

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, knnIndexSettings, builder.toString());
        // Disable refresh
        updateIndexSettings(indexName, Settings.builder().put("index.refresh_interval", -1));

        // Index the test data without refresh on every document
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName),
                ImmutableList.of(Floats.asList(testData.indexData.vectors[i]).toArray()),
                false
            );
        }

        refreshAllIndices();
        // Assert we have the right number of documents in the index
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        final int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            final Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, testData.queries[i], k), k);
            final String responseBody = EntityUtils.toString(response.getEntity());
            final List<KNNResult> nmslibValidNeighbors = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, nmslibValidNeighbors.size());
        }
        // Delete index
        deleteKNNIndex(indexName);
    }

    /*
      For this testcase, we will create index with setting build_vector_data_structure_threshold as -1, then index few documents, perform knn search,
      then, confirm hits because of exact search though there are no graph. In next step, update setting to 0, force merge segment to 1, perform knn search and confirm expected
      hits are returned.
    */
    public void testKNNIndex_whenBuildVectorGraphThresholdIsProvidedEndToEnd_thenBuildGraphBasedOnSettingUsingRadialSearch()
        throws Exception {
        final String indexName = "test-index-1";
        final String fieldName1 = "test-field-1";
        final String fieldName2 = "test-field-2";

        final Integer dimension = testData.indexData.vectors[0].length;
        final Settings knnIndexSettings = buildKNNIndexSettings(-1);

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName1)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .startObject(fieldName2)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(KNNConstants.PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, knnIndexSettings, builder.toString());

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName1, fieldName2),
                ImmutableList.of(
                    Floats.asList(testData.indexData.vectors[i]).toArray(),
                    Floats.asList(testData.indexData.vectors[i]).toArray()
                )
            );
        }

        refreshAllIndices();
        // Assert we have the right number of documents in the index
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        final List<KNNResult> nmslibNeighbors = getResults(indexName, fieldName1, testData.queries[0], 1);
        assertEquals("unexpected neighbors are returned", nmslibNeighbors.size(), nmslibNeighbors.size());

        final List<KNNResult> faissNeighbors = getResults(indexName, fieldName2, testData.queries[0], 1);
        assertEquals("unexpected neighbors are returned", faissNeighbors.size(), faissNeighbors.size());

        // update build vector data structure setting
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0));
        forceMergeKnnIndex(indexName, 1);

        final int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            // Search nmslib field
            final Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName1, testData.queries[i], k), k);
            final String responseBody = EntityUtils.toString(response.getEntity());
            final List<KNNResult> nmslibValidNeighbors = parseSearchResponse(responseBody, fieldName1);
            assertEquals(k, nmslibValidNeighbors.size());
            // Search faiss field
            final List<KNNResult> faissValidNeighbors = getResults(indexName, fieldName2, testData.queries[i], k);
            assertEquals(k, faissValidNeighbors.size());
        }

        // Delete index
        deleteKNNIndex(indexName);
    }

    private List<KNNResult> getResults(final String indexName, final String fieldName, final float[] vector, final int k)
        throws IOException, ParseException {
        final Response searchResponseField = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, vector, k), k);
        final String searchResponseBody = EntityUtils.toString(searchResponseField.getEntity());
        return parseSearchResponse(searchResponseBody, fieldName);
    }

}
