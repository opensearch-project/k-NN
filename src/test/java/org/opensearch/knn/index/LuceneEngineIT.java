/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.After;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.Strings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class LuceneEngineIT extends KNNRestTestCase {

    private static final int DIMENSION = 3;
    private static final String DOC_ID = "doc1";
    private static final String DOC_ID_2 = "doc2";
    private static final String DOC_ID_3 = "doc3";
    private static final int EF_CONSTRUCTION = 128;
    private static final String COLOR_FIELD_NAME = "color";
    private static final String TASTE_FIELD_NAME = "taste";
    private static final int M = 16;

    private static final Float[][] TEST_INDEX_VECTORS = { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };

    private static final float[][] TEST_QUERY_VECTORS = { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };

    private static final Map<VectorSimilarityFunction, Function<Float, Float>> VECTOR_SIMILARITY_TO_SCORE = ImmutableMap.of(
        VectorSimilarityFunction.EUCLIDEAN,
        (similarity) -> 1 / (1 + similarity),
        VectorSimilarityFunction.DOT_PRODUCT,
        (similarity) -> (1 + similarity) / 2,
        VectorSimilarityFunction.COSINE,
        (similarity) -> (1 + similarity) / 2
    );
    private static final String DIMENSION_FIELD_NAME = "dimension";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String PROPERTIES_FIELD_NAME = "properties";
    private static final String TYPE_FIELD_NAME = "type";

    @After
    public final void cleanUp() throws IOException {
        deleteKNNIndex(INDEX_NAME);
    }

    public void testQuery_l2() throws Exception {
        baseQueryTest(SpaceType.L2);
    }

    public void testQuery_cosine() throws Exception {
        baseQueryTest(SpaceType.COSINESIMIL);
    }

    public void testQuery_innerProduct_notSupported() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.LUCENE.getMethod(METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = Strings.toString(builder);

        createIndex(INDEX_NAME, getKNNDefaultIndexSettings());

        Request request = new Request("PUT", "/" + INDEX_NAME + "/_mapping");
        request.setJsonEntity(mapping);

        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }

    public void testQuery_invalidVectorDimensionInQuery() throws Exception {

        createKnnIndexMappingWithLuceneEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        float[] invalidQuery = new float[DIMENSION - 1];
        int validK = 1;
        expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, invalidQuery, validK), validK)
        );
    }

    public void testQuery_documentsMissingField() throws Exception {

        SpaceType spaceType = SpaceType.L2;

        createKnnIndexMappingWithLuceneEngine(DIMENSION, spaceType, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        // Add a doc without the lucene field set
        String secondField = "field-2";
        addDocWithNumericField(INDEX_NAME, Integer.toString(TEST_INDEX_VECTORS.length + 1), secondField, 0L);

        validateQueries(spaceType, FIELD_NAME);
    }

    public void testQuery_multipleEngines() throws Exception {
        String luceneField = "lucene-field";
        SpaceType luceneSpaceType = SpaceType.COSINESIMIL;
        String nmslibField = "nmslib-field";
        SpaceType nmslibSpaceType = SpaceType.L2;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(luceneField)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, luceneSpaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .startObject(nmslibField)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, nmslibSpaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        String mapping = Strings.toString(builder);
        createKnnIndex(INDEX_NAME, mapping);

        for (int i = 0; i < TEST_INDEX_VECTORS.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(i + 1),
                ImmutableList.of(luceneField, nmslibField),
                ImmutableList.of(TEST_INDEX_VECTORS[i], TEST_INDEX_VECTORS[i])
            );
        }

        validateQueries(luceneSpaceType, luceneField);
        validateQueries(nmslibSpaceType, nmslibField);
    }

    public void testAddDoc() throws Exception {
        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.LUCENE.getMethod(METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = Strings.toString(builder);

        createKnnIndex(INDEX_NAME, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        Float[] vector = new Float[] { 2.0f, 4.5f, 6.5f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    public void testUpdateDoc() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        Float[] updatedVector = { 8.0f, 8.0f };
        updateKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, updatedVector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    public void testDeleteDoc() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        deleteKnnDoc(INDEX_NAME, DOC_ID);

        refreshAllIndices();
        assertEquals(0, getDocCount(INDEX_NAME));
    }

    public void testQueryWithFilterUsingFloatVectorDataType() throws Exception {
        createKnnIndexMappingWithLuceneEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

        addKnnDocWithAttributes(
            DOC_ID,
            new float[] { 6.0f, 7.9f, 3.1f },
            ImmutableMap.of(COLOR_FIELD_NAME, "red", TASTE_FIELD_NAME, "sweet")
        );
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        refreshAllIndices();

        final float[] searchVector = { 6.0f, 6.0f, 4.1f };
        List<String> expectedDocIdsKGreaterThanFilterResult = Arrays.asList(DOC_ID, DOC_ID_3);
        List<String> expectedDocIdsKLimitsFilterResult = Arrays.asList(DOC_ID);
        validateQueryResultsWithFilters(searchVector, 5, 1, expectedDocIdsKGreaterThanFilterResult, expectedDocIdsKLimitsFilterResult);
    }

    @SneakyThrows
    public void testQueryWithFilterUsingByteVectorDataType() {
        createKnnIndexMappingWithLuceneEngine(3, SpaceType.L2, VectorDataType.BYTE);

        addKnnDocWithAttributes(DOC_ID, new float[] { 6.0f, 7.0f, 3.0f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.0f, 2.0f, 4.0f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.0f, 5.0f, 7.0f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        refreshAllIndices();

        final float[] searchVector = { 6.0f, 6.0f, 4.0f };
        List<String> expectedDocIdsKGreaterThanFilterResult = Arrays.asList(DOC_ID, DOC_ID_3);
        List<String> expectedDocIdsKLimitsFilterResult = Arrays.asList(DOC_ID);
        validateQueryResultsWithFilters(searchVector, 5, 1, expectedDocIdsKGreaterThanFilterResult, expectedDocIdsKLimitsFilterResult);
    }

    public void testQuery_filterWithNonLuceneEngine() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, NMSLIB_NAME)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = Strings.toString(builder);
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDocWithAttributes(
            DOC_ID,
            new float[] { 6.0f, 7.9f, 3.1f },
            ImmutableMap.of(COLOR_FIELD_NAME, "red", TASTE_FIELD_NAME, "sweet")
        );
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        final float[] searchVector = { 6.0f, 6.0f, 5.6f };
        int k = 5;
        expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(
                INDEX_NAME,
                new KNNQueryBuilder(FIELD_NAME, searchVector, k, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
                k
            )
        );
    }

    public void testIndexReopening() throws Exception {
        createKnnIndexMappingWithLuceneEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        final float[] searchVector = TEST_QUERY_VECTORS[0];
        final int k = 1 + RandomUtils.nextInt(TEST_INDEX_VECTORS.length);

        final List<Float[]> knnResultsBeforeIndexClosure = queryResults(searchVector, k);

        closeIndex(INDEX_NAME);
        openIndex(INDEX_NAME);

        ensureGreen(INDEX_NAME);

        final List<Float[]> knnResultsAfterIndexClosure = queryResults(searchVector, k);

        assertArrayEquals(knnResultsBeforeIndexClosure.toArray(), knnResultsAfterIndexClosure.toArray());
    }

    private void createKnnIndexMappingWithLuceneEngine(int dimension, SpaceType spaceType, VectorDataType vectorDataType) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.LUCENE.getMethod(METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = Strings.toString(builder);
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void baseQueryTest(SpaceType spaceType) throws Exception {

        createKnnIndexMappingWithLuceneEngine(DIMENSION, spaceType, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        validateQueries(spaceType, FIELD_NAME);
    }

    private void validateQueries(SpaceType spaceType, String fieldName) throws Exception {

        int k = LuceneEngineIT.TEST_INDEX_VECTORS.length;
        for (float[] queryVector : TEST_QUERY_VECTORS) {
            Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(fieldName, queryVector, k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
                float distance = TestUtils.computeDistFromSpaceType(spaceType, primitiveArray, queryVector);
                float rawScore = VECTOR_SIMILARITY_TO_SCORE.get(spaceType.getVectorSimilarityFunction()).apply(distance);
                assertEquals(KNNEngine.LUCENE.score(rawScore, spaceType), actualScores.get(j), 0.0001);
            }
        }
    }

    private List<Float[]> queryResults(final float[] searchVector, final int k) throws Exception {
        final String responseBody = EntityUtils.toString(
            searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, searchVector, k), k).getEntity()
        );
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertNotNull(knnResults);
        return knnResults.stream().map(KNNResult::getVector).collect(Collectors.toUnmodifiableList());
    }

    @SneakyThrows
    private void validateQueryResultsWithFilters(
        float[] searchVector,
        int kGreaterThanFilterResult,
        int kLimitsFilterResult,
        List<String> expectedDocIdsKGreaterThanFilterResult,
        List<String> expectedDocIdsKLimitsFilterResult
    ) {
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kGreaterThanFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kGreaterThanFilterResult
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(expectedDocIdsKGreaterThanFilterResult.size(), knnResults.size());
        assertTrue(
            knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toList()).containsAll(expectedDocIdsKGreaterThanFilterResult)
        );

        final Response responseKLimitsFilterResult = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kLimitsFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kLimitsFilterResult
        );
        final String responseBodyKLimitsFilterResult = EntityUtils.toString(responseKLimitsFilterResult.getEntity());
        final List<KNNResult> knnResultsKLimitsFilterResult = parseSearchResponse(responseBodyKLimitsFilterResult, FIELD_NAME);

        assertEquals(expectedDocIdsKLimitsFilterResult.size(), knnResultsKLimitsFilterResult.size());
        assertTrue(
            knnResultsKLimitsFilterResult.stream()
                .map(KNNResult::getDocId)
                .collect(Collectors.toList())
                .containsAll(expectedDocIdsKLimitsFilterResult)
        );
    }
}
