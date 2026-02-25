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

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.*;
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
