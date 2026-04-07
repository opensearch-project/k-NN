/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.extension;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.search.processor.mmr.MMROverSampleProcessor;
import org.opensearch.knn.search.processor.mmr.MMRRerankProcessor;

import org.opensearch.core.xcontent.MediaTypeRegistry;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.opensearch.knn.search.processor.mmr.MMRExplainInfo;

import static org.opensearch.knn.KNNRestTestCase.PROPERTIES_FIELD;
import static org.opensearch.knn.common.KNNConstants.CANDIDATES;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.DIVERSITY;
import static org.opensearch.knn.common.KNNConstants.EXPLAIN;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MMR;
import static org.opensearch.knn.common.KNNConstants.MMR_EXPLAIN;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_FIELD_PATH;
import static org.opensearch.knn.common.KNNConstants.VECTOR_FIELD_SPACE_TYPE;
import static org.opensearch.search.pipeline.SearchPipelineService.ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING;

public class MMRSearchExtBuilderIT extends KNNRestTestCase {
    private static final int DIMENSION_NUM = 2;
    private static final String FIELD_NAME = "vector_field";
    private static final String INDEX_NAME = "test_index";
    private static final int QUERY_SIZE = 3;
    private static final float[] SIMILAR_VECTOR = new float[] { 1f, 1f };
    private static final float[] DIVERSE_VECTOR_1 = new float[] { 1f, 2f };
    private static final float[] DIVERSE_VECTOR_2 = new float[] { 2f, 1f };
    private float DELTA = 1e-6F;

    @Before
    public void setUpForMMR() {
        enableMMRProcessors();
        createTestIndexAndDocs();
    }

    @After
    public void cleanUpForMMR() throws IOException {
        deleteKNNIndex(INDEX_NAME);
        disableMMRProcessors();
    }

    @SneakyThrows
    public void testMMR_whenRerankWithVectors_thenSelectTop3() {
        XContentBuilder queryBuilder = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, false, false);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        verifyResults(results, false);
    }

    @SneakyThrows
    public void testMMR_whenSourceExcludesVector_thenVectorExcluded() {
        XContentBuilder queryBuilder = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, true, false);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        verifyResults(results, true);
    }

    @SneakyThrows
    public void testMMR_whenDisabledStoredFields_thenVectorExcluded() {
        XContentBuilder queryBuilderDisabledStoredFields = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, false, false, "_none_");

        Response response = searchKNNIndex(INDEX_NAME, queryBuilderDisabledStoredFields.toString(), QUERY_SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        verifyResults(results, true);
    }

    @SneakyThrows
    public void testMMR_whenEmptyStoredFieldsAndExplicitlyEnableSource_thenVectorIncluded() {
        XContentBuilder queryBuilderDisabledStoredFields = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, true, false, "empty");

        Response response = searchKNNIndex(INDEX_NAME, queryBuilderDisabledStoredFields.toString(), QUERY_SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        verifyResults(results, true);
    }

    @SneakyThrows
    public void testMMR_whenUserProvidedVectorPath_thenVectorIncluded() {
        XContentBuilder queryBuilder = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, false, true);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        verifyResults(results, false);
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testMMR_whenExplainEnabled_thenExplainInfoInResponse() {
        XContentBuilder queryBuilder = buildMMRQueryWithExplain(SIMILAR_VECTOR, QUERY_SIZE, true);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(QUERY_SIZE, results.size());

        // Parse the raw response to check for mmr_explain in _source
        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        Map<String, Object> hitsOuter = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hitsOuter.get("hits");

        for (int i = 0; i < hitsList.size(); i++) {
            Map<String, Object> hit = hitsList.get(i);
            Map<String, Object> source = (Map<String, Object>) hit.get("_source");
            assertNotNull("Hit " + i + " should have _source", source);

            Map<String, Object> mmrExplain = (Map<String, Object>) source.get(MMR_EXPLAIN);
            assertNotNull("Hit " + i + " should have mmr_explain in _source", mmrExplain);
            assertNotNull("original_score should be present", mmrExplain.get(MMRExplainInfo.ORIGINAL_SCORE_FIELD));
            assertNotNull("max_similarity_to_selected should be present", mmrExplain.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD));
            assertNotNull("mmr_score should be present", mmrExplain.get(MMRExplainInfo.MMR_SCORE_FIELD));
            assertNotNull("mmr_formula should be present", mmrExplain.get(MMRExplainInfo.MMR_FORMULA_FIELD));
            // selection_round and selected_so_far are intentionally absent - inferred from result position
            assertNull("selection_round should NOT be present", mmrExplain.get("selection_round"));
            assertNull("selected_so_far should NOT be present", mmrExplain.get("selected_so_far"));
        }

        // Verify first hit has max_similarity_to_selected = 0 (no prior selections)
        Map<String, Object> firstSource = (Map<String, Object>) hitsList.get(0).get("_source");
        Map<String, Object> firstExplain = (Map<String, Object>) firstSource.get(MMR_EXPLAIN);
        assertEquals(0.0, ((Number) firstExplain.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD)).doubleValue(), 1e-6);
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testMMR_whenExplainDisabled_thenNoExplainInfoInResponse() {
        XContentBuilder queryBuilder = buildMMRQueryWithExplain(SIMILAR_VECTOR, QUERY_SIZE, false);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(QUERY_SIZE, results.size());

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        Map<String, Object> hitsOuter = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hitsOuter.get("hits");

        for (int i = 0; i < hitsList.size(); i++) {
            Map<String, Object> hit = hitsList.get(i);
            Map<String, Object> source = (Map<String, Object>) hit.get("_source");
            assertNotNull("Hit " + i + " should have _source", source);
            assertNull("Hit " + i + " should NOT have mmr_explain when explain is disabled", source.get(MMR_EXPLAIN));
        }
    }

    @SuppressWarnings("unchecked")
    @SneakyThrows
    public void testMMR_whenExplainNotSpecified_thenNoExplainInfoInResponse() {
        // Use the standard query builder without explain field
        XContentBuilder queryBuilder = buildMMRQuery(SIMILAR_VECTOR, QUERY_SIZE, false, false);

        Response response = searchKNNIndex(INDEX_NAME, queryBuilder.toString(), QUERY_SIZE);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(QUERY_SIZE, results.size());

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        Map<String, Object> hitsOuter = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hitsOuter.get("hits");

        for (int i = 0; i < hitsList.size(); i++) {
            Map<String, Object> hit = hitsList.get(i);
            Map<String, Object> source = (Map<String, Object>) hit.get("_source");
            assertNotNull("Hit " + i + " should have _source", source);
            assertNull("Hit " + i + " should NOT have mmr_explain when explain is not specified", source.get(MMR_EXPLAIN));
        }
    }

    @SneakyThrows
    private void enableMMRProcessors() {
        updateClusterSettings(
            ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey(),
            new String[] { MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE, MMRRerankProcessor.MMRRerankProcessorFactory.TYPE }
        );
    }

    @SneakyThrows
    private void disableMMRProcessors() {
        updateClusterSettings(ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey(), "");
    }

    @SneakyThrows
    private void createTestIndexAndDocs() {
        XContentBuilder mappingBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, DIMENSION_NUM)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, mappingBuilder.toString(), 3);

        for (int i = 0; i < 8; i++)
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, SIMILAR_VECTOR);

        float[][] diverseVectors = new float[][] { DIVERSE_VECTOR_1, DIVERSE_VECTOR_2 };
        for (int i = 8; i < 10; i++)
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, diverseVectors[i - 8]);
    }

    @SneakyThrows
    private XContentBuilder buildMMRQuery(float[] queryVector, int k, boolean excludeVector, boolean userProvidedVector) {
        return buildMMRQuery(queryVector, k, excludeVector, userProvidedVector, null);
    }

    @SneakyThrows
    private XContentBuilder buildMMRQuery(
        float[] queryVector,
        int k,
        boolean excludeVector,
        boolean userProvidedVector,
        String storedFields
    ) {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        if ("_none_".equals(storedFields)) {
            builder.field("stored_fields", "_none_");
        } else if ("empty".equals(storedFields)) {
            builder.array("stored_fields", "");
        }
        if (excludeVector) {
            builder.startObject("_source").array("excludes", FIELD_NAME).endObject();
        } else if ("empty".equals(storedFields)) {
            builder.field("_source", true);
        }

        builder.startObject("query")
            .startObject(KNN)
            .startObject(FIELD_NAME)
            .array(VECTOR, queryVector)
            .field(K, k)
            .endObject()
            .endObject()
            .endObject();

        builder.startObject("ext").startObject(MMR).field(CANDIDATES, 9).field(DIVERSITY, 0.9);
        if (userProvidedVector) {
            builder.field(VECTOR_FIELD_PATH, FIELD_NAME).field(VECTOR_FIELD_SPACE_TYPE, SpaceType.L2.getValue());
        }
        builder.endObject().endObject().endObject();

        return builder;
    }

    @SneakyThrows
    private XContentBuilder buildMMRQueryWithExplain(float[] queryVector, int k, boolean explain) {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        builder.startObject("query")
            .startObject(KNN)
            .startObject(FIELD_NAME)
            .array(VECTOR, queryVector)
            .field(K, k)
            .endObject()
            .endObject()
            .endObject();

        builder.startObject("ext")
            .startObject(MMR)
            .field(CANDIDATES, 9)
            .field(DIVERSITY, 0.9)
            .field(EXPLAIN, explain)
            .endObject()
            .endObject()
            .endObject();

        return builder;
    }

    private void verifyResults(List<KNNResult> results, boolean excludeVector) {
        if (excludeVector) {
            results.forEach(r -> assertNull("Vector should be excluded", r.getVector()));
        } else {
            results.forEach(r -> assertNotNull("Vector should be included", r.getVector()));
        }
        assertEquals(QUERY_SIZE, results.size());

        if (results.get(0).getVector() != null) {
            assertArrayEquals(SIMILAR_VECTOR, results.get(0).getVector(), DELTA);
        }
        assertEquals(1.0, (double) results.get(0).getScore(), DELTA);
        assertEquals("Should pick up the hit with diversity.", "8", results.get(1).getDocId());
        assertEquals(0.5, (double) results.get(1).getScore(), DELTA);
        if (results.get(1).getVector() != null) {
            assertArrayEquals("Should pick up the hit with diversity.", DIVERSE_VECTOR_1, results.get(1).getVector(), DELTA);
        }
        if (results.get(2).getVector() != null) {
            assertArrayEquals(SIMILAR_VECTOR, results.get(2).getVector(), DELTA);
        }
        assertEquals(1.0, (double) results.get(2).getScore(), DELTA);
    }
}
