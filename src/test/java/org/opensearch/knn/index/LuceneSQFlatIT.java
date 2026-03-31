/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.apache.lucene.tests.util.LuceneTestCase.expectThrows;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;

public class LuceneSQFlatIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final String PROPERTIES_FIELD = "properties";
    private static final String TYPE_FIELD = "type";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String DIMENSION_FIELD = "dimension";
    private static final String COLOR_FIELD_NAME = "color";
    private static final String NESTED_FIELD_NAME = "nested_field";
    private static final String NESTED_VECTOR_FIELD = "nested_vector";

    @After
    public final void cleanUp() throws IOException {
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexAndQuery_withFlatMethod_l2() {
        createFlatIndex(SpaceType.L2);
        indexTestDocs();

        // Query with all 1s — closest docs by L2: doc0(all 1s), doc1(all 2s), doc2(all 3s)
        float[] queryVector = generateVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("0", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());

        List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
        for (int i = 0; i < scores.size() - 1; i++) {
            assertTrue("Scores should be in descending order", scores.get(i) >= scores.get(i + 1));
        }
    }

    @SneakyThrows
    public void testIndexAndQuery_withFlatMethod_cosine() {
        createFlatIndex(SpaceType.COSINESIMIL);
        indexTestDocs();

        float[] queryVector = generateVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());

        // All constant vectors point in the same direction, so cosine scores should all be close to max
        List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
        for (Float score : scores) {
            assertTrue("Cosine score should be positive", score > 0);
        }
    }

    @SneakyThrows
    public void testIndexAndQuery_withFlatMethod_innerProduct() {
        createFlatIndex(SpaceType.INNER_PRODUCT);
        indexTestDocs();

        // Query with all 1s — highest inner product: doc4(all 5s), doc3(all 4s), doc2(all 3s)
        float[] queryVector = generateVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("4", results.get(0).getDocId());
        assertEquals("3", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());

        List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
        for (int i = 0; i < scores.size() - 1; i++) {
            assertTrue("Scores should be in descending order", scores.get(i) >= scores.get(i + 1));
        }
    }

    @SneakyThrows
    public void testFlatMethod_withFaissEngine_thenFail() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, builder.toString()));
    }

    @SneakyThrows
    public void testFlatMethod_withParameters_thenFail() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, 16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, builder.toString()));
    }

    @SneakyThrows
    public void testFlatMethod_withUnsupportedCompression_thenFail() {
        String[] unsupportedCompressions = { "1x", "2x", "4x", "8x", "16x", "64x" };
        for (String compression : unsupportedCompressions) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(FIELD_NAME)
                .field(TYPE_FIELD, KNN_VECTOR_TYPE)
                .field(DIMENSION_FIELD, DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compression)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, METHOD_FLAT)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            expectThrows(
                ResponseException.class,
                String.format(Locale.ROOT, "Expected failure for compression level %s", compression),
                () -> createKnnIndex(INDEX_NAME, builder.toString())
            );
        }
    }

    @SneakyThrows
    public void testFlatMethod_withMode_thenFail() {
        String[] modes = { "on_disk", "in_memory" };
        for (String mode : modes) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(FIELD_NAME)
                .field(TYPE_FIELD, KNN_VECTOR_TYPE)
                .field(DIMENSION_FIELD, DIMENSION)
                .field(MODE_PARAMETER, mode)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, METHOD_FLAT)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            expectThrows(
                ResponseException.class,
                String.format(Locale.ROOT, "Expected failure for mode %s", mode),
                () -> createKnnIndex(INDEX_NAME, builder.toString())
            );
        }
    }

    @SneakyThrows
    public void testFlatMethod_withDeleteAndUpdate() {
        createFlatIndex(SpaceType.L2);

        Float[] vector1 = new Float[DIMENSION];
        Float[] vector2 = new Float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector1[i] = 1.0f;
            vector2[i] = 2.0f;
        }

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector1);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector2);
        refreshIndex(INDEX_NAME);
        assertEquals(2, getDocCount(INDEX_NAME));

        // Update doc
        updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector2);
        refreshIndex(INDEX_NAME);
        assertEquals(2, getDocCount(INDEX_NAME));

        // Delete doc
        deleteKnnDoc(INDEX_NAME, "2");
        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));

        // Search should still work — only doc "1" remains (updated to all 2s)
        float[] queryVector = generateVector(DIMENSION, 2.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 1), 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(1, results.size());
        assertEquals("1", results.get(0).getDocId());
        List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertTrue("Score should be positive", scores.get(0) > 0);
    }

    @SneakyThrows
    public void testFlatMethod_withFilter() {
        // Create index with flat method and an additional keyword field
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .startObject(COLOR_FIELD_NAME)
            .field(TYPE_FIELD, "keyword")
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());

        // Index docs: 3 red, 2 green
        addKnnDocWithAttributes("0", generateVector(DIMENSION, 1.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        addKnnDocWithAttributes("1", generateVector(DIMENSION, 2.0f), ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes("2", generateVector(DIMENSION, 3.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        addKnnDocWithAttributes("3", generateVector(DIMENSION, 4.0f), ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes("4", generateVector(DIMENSION, 5.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        refreshIndex(INDEX_NAME);

        // Query with all 1s, filter to red only — expect docs 0, 2, 4
        float[] queryVector = generateVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, queryVector, 5, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            5
        );
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());
        // Nearest red doc to all-1s query is doc 0
        assertEquals("0", results.get(0).getDocId());

        // Verify no green docs in results
        for (KNNResult result : results) {
            assertTrue(
                "Only red docs expected",
                result.getDocId().equals("0") || result.getDocId().equals("2") || result.getDocId().equals("4")
            );
        }

        List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
        for (int i = 0; i < scores.size() - 1; i++) {
            assertTrue("Scores should be in descending order", scores.get(i) >= scores.get(i + 1));
        }
    }

    @SneakyThrows
    public void testFlatMethod_withNestedField() {
        // Create index with nested field containing a flat vector
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_FIELD_NAME)
            .field(TYPE_FIELD, "nested")
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_VECTOR_FIELD)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());

        // Index 3 docs, each with 2 nested vectors
        // Doc 0: vectors at [1,1,...] and [2,2,...]
        // Doc 1: vectors at [3,3,...] and [4,4,...]
        // Doc 2: vectors at [5,5,...] and [6,6,...]
        for (int i = 0; i < 3; i++) {
            Float[] vectorA = new Float[DIMENSION];
            Float[] vectorB = new Float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                vectorA[d] = (float) (i * 2 + 1);
                vectorB[d] = (float) (i * 2 + 2);
            }
            String doc = NestedKnnDocBuilder.create(NESTED_FIELD_NAME).addVectors(NESTED_VECTOR_FIELD, vectorA, vectorB).build();
            addKnnDoc(INDEX_NAME, Integer.toString(i), doc);
        }
        refreshIndex(INDEX_NAME);

        // Query with all-1s — doc 0 has the closest nested vector [1,1,...]
        Float[] queryVector = new Float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            queryVector[i] = 1.0f;
        }
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(responseBody));
        List<String> docIds = parseIds(responseBody);
        assertEquals("0", docIds.get(0));
        assertEquals("1", docIds.get(1));
    }

    private void createFlatIndex(SpaceType spaceType) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());
    }

    @SneakyThrows
    public void testExpandNestedDocs_withFlatMethod() {
        int numberOfNestedVectors = 2;
        int numberOfDocs = 3;
        int k = numberOfDocs;

        // Create nested index with flat method
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_FIELD_NAME)
            .field(TYPE_FIELD, "nested")
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_VECTOR_FIELD)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_FLAT)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());

        // Index docs, each with numberOfNestedVectors nested vectors
        for (int i = 0; i < numberOfDocs; i++) {
            NestedKnnDocBuilder docBuilder = NestedKnnDocBuilder.create(NESTED_FIELD_NAME);
            for (int j = 0; j < numberOfNestedVectors; j++) {
                Float[] vector = new Float[DIMENSION];
                for (int d = 0; d < DIMENSION; d++) {
                    vector[d] = (float) (i * numberOfNestedVectors + j + 1);
                }
                docBuilder.addVectors(NESTED_VECTOR_FIELD, vector);
            }
            addKnnDoc(INDEX_NAME, Integer.toString(i), docBuilder.build());
        }
        refreshIndex(INDEX_NAME);

        // Query with expand_nested_docs=true — expect all nested docs per parent returned via inner_hits
        Float[] queryVector = new Float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            queryVector[d] = 1.0f;
        }
        Response response = queryNestedFieldWithExpandNestedDocs(INDEX_NAME, k, queryVector);
        String entity = EntityUtils.toString(response.getEntity());

        Multimap<String, Integer> docIdToOffsets = parseInnerHits(entity, NESTED_FIELD_NAME);
        assertEquals(numberOfDocs, docIdToOffsets.keySet().size());
        for (String docId : docIdToOffsets.keySet()) {
            assertEquals(
                "Each parent doc should have all nested vectors in inner_hits",
                numberOfNestedVectors,
                docIdToOffsets.get(docId).size()
            );
        }
    }

    private Response queryNestedFieldWithExpandNestedDocs(final String index, final int k, final Float[] vector) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(QUERY)
            .startObject(TYPE_NESTED)
            .field(PATH, NESTED_FIELD_NAME)
            .startObject(QUERY)
            .startObject(KNN)
            .startObject(NESTED_FIELD_NAME + "." + NESTED_VECTOR_FIELD)
            .field(VECTOR, vector)
            .field(K, k)
            .field(EXPAND_NESTED, true)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return response;
    }

    @SneakyThrows
    public void testRadialSearch_withMinScore() {
        createFlatIndex(SpaceType.L2);
        indexTestDocs();

        // L2 scores from all-1s query: doc0(all 1s)=1.0, doc1(all 2s)=1/(1+128), doc2..doc4 even lower
        // Only doc0 should have score >= 0.9
        float minScore = 0.9f;
        int[] expectedResultCounts = { 1, 1, 1 };
        float[][] queryVectors = { generateVector(DIMENSION, 1.0f), generateVector(DIMENSION, 1.0f), generateVector(DIMENSION, 1.0f) };

        validateRadiusSearchResults(queryVectors, null, minScore, expectedResultCounts);
    }

    @SneakyThrows
    public void testRadialSearch_withMaxDistance() {
        createFlatIndex(SpaceType.L2);
        indexTestDocs();

        // L2 squared distances from all-1s query: doc0=0, doc1=128*(2-1)^2=128, doc2=128*(3-1)^2=512, ...
        // Only doc0 and doc1 should be within maxDistance=200
        float maxDistance = 200.0f;
        int[] expectedResultCounts = { 2, 2, 2 };
        float[][] queryVectors = { generateVector(DIMENSION, 1.0f), generateVector(DIMENSION, 1.0f), generateVector(DIMENSION, 1.0f) };

        validateRadiusSearchResults(queryVectors, maxDistance, null, expectedResultCounts);
    }

    private void validateRadiusSearchResults(
        final float[][] queryVectors,
        final Float maxDistance,
        final Float minScore,
        final int[] expectedResultCounts
    ) throws Exception {
        for (int i = 0; i < queryVectors.length; i++) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", queryVectors[i]);
            if (maxDistance != null) {
                builder.field(MAX_DISTANCE, maxDistance);
            } else if (minScore != null) {
                builder.field(MIN_SCORE, minScore);
            } else {
                throw new IllegalArgumentException("Either maxDistance or minScore must be provided");
            }
            builder.endObject().endObject().endObject().endObject();

            String responseBody = EntityUtils.toString(searchKNNIndex(INDEX_NAME, builder, expectedResultCounts[i]).getEntity());
            List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
            assertEquals(expectedResultCounts[i], results.size());

            List<Float> scores = parseSearchResponseScore(responseBody, FIELD_NAME);
            for (int j = 0; j < results.size(); j++) {
                if (minScore != null) {
                    assertTrue(
                        String.format(Locale.ROOT, "Score %.4f should be >= minScore %.4f", scores.get(j), minScore),
                        scores.get(j) >= minScore - 0.001f
                    );
                }
                if (maxDistance != null) {
                    float score = scores.get(j);
                    // L2 score = 1 / (1 + distance), so distance = (1/score) - 1
                    float distance = (1.0f / score) - 1.0f;
                    assertTrue(
                        String.format(Locale.ROOT, "Distance %.4f should be <= maxDistance %.4f", distance, maxDistance),
                        distance <= maxDistance + 0.001f
                    );
                }
            }
        }
    }

    private void indexTestDocs() throws Exception {
        for (int i = 0; i < 5; i++) {
            Float[] vector = new Float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                vector[d] = (float) (i + 1);
            }
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }
        refreshIndex(INDEX_NAME);
        assertEquals(5, getDocCount(INDEX_NAME));
    }

    private float[] generateVector(int dimension, float value) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = value;
        }
        return vector;
    }

    private Response queryNestedField(String index, int k, Float[] vector) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", NESTED_FIELD_NAME)
            .startObject("query")
            .startObject("knn")
            .startObject(NESTED_FIELD_NAME + "." + NESTED_VECTOR_FIELD)
            .field("vector", vector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return response;
    }
}
