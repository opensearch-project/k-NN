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
import org.junit.BeforeClass;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.Strings;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.containsString;

public class OpenSearchIT extends KNNRestTestCase {

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        URL testIndexVectors = FaissIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = FaissIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
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

        KNNMethod method1 = knnEngine1.getMethod(KNNConstants.METHOD_HNSW);
        KNNMethod method2 = knnEngine2.getMethod(KNNConstants.METHOD_HNSW);
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
            .field(KNNConstants.NAME, method1.getMethodComponent().getName())
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
            .field(KNNConstants.NAME, method2.getMethodComponent().getName())
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
        String mapping = Strings.toString(builder);
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
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
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
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
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
        String expMessage = "Indexing knn vector fields is rejected as circuit breaker triggered."
            + " Check _opendistro/_knn/stats for detailed state";
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
        String expMessage = "Indexing knn vector fields is rejected as circuit breaker triggered."
            + " Check _opendistro/_knn/stats for detailed state";
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

    public void testVectorMappingValidation_noDimension() throws Exception {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        String mapping = Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .endObject()
                .endObject()
                .endObject()
        );

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
                    + " for vector: "
                    + FIELD_NAME
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
        String mapping = Strings.toString(
            XContentFactory.jsonBuilder()
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
        );

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
        request.setJsonEntity(Strings.toString(builder));
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

}
