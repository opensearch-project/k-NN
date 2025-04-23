/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.RescoreParser;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ANN_SEARCH;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DISK_BASED_SEARCH;
import static org.opensearch.knn.common.KNNConstants.EXACT_SEARCH;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.RADIAL_SEARCH;

public class ExplainIT extends KNNRestTestCase {

    @SneakyThrows
    public void testExplain_whenDefault_thenANNSearch() {
        int dimension = 128;
        int numDocs = 100;
        createDefaultKnnIndex(dimension);
        indexTestData(INDEX_NAME, FIELD_NAME, dimension, numDocs);
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);
        XContentBuilder queryBuilder = buildSearchQuery(FIELD_NAME, 10, queryVector, null);
        // validate primaries are working
        validateExplainSearchResponse(
            queryBuilder,
            ANN_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            SpaceType.L2.explainScoreTranslation(0)
        );
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testExplain_whenFilterIdLessThanK_thenANNWithExactSearch() {
        createDefaultKnnIndex(2);
        indexTestData(INDEX_NAME, FIELD_NAME, 2, 2);

        // Execute the search request with a match all query to ensure exact logic gets called
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 1000));

        float[] queryVector = new float[] { 1.0f, 1.0f };

        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, 2, QueryBuilders.matchAllQuery());
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(queryBuilder, ToXContent.EMPTY_PARAMS);
        queryBuilder.endObject().endObject();

        validateExplainSearchResponse(
            queryBuilder,
            ANN_SEARCH,
            EXACT_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            "since filteredIds",
            "is less than or equal to K"
        );
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testExplain_whenDefaultWithDist_thenRadialWithANNSearch() {
        int dimension = 128;
        int numDocs = 100;
        createDefaultKnnIndex(dimension);
        indexTestData(INDEX_NAME, FIELD_NAME, dimension, numDocs);
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        float distance = 15f;
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field(MAX_DISTANCE, distance)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        validateExplainSearchResponse(
            queryBuilder,
            RADIAL_SEARCH,
            ANN_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            SpaceType.L2.explainScoreTranslation(0),
            String.valueOf(distance)
        );
        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testExplain_whenFilerQueryWithDist_thenRadialWithExactSearch() {
        setupKNNIndexForFilterQuery();

        final float[] queryVector = new float[] { 3.3f, 3.0f, 5.0f };
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("color", "red");
        float distance = 15f;

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field(MAX_DISTANCE, distance)
            .field("filter", termQueryBuilder)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        validateExplainSearchResponse(
            queryBuilder,
            RADIAL_SEARCH,
            EXACT_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            String.valueOf(distance)
        );

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testExplain_whenDefaultDiskBasedSearch_thenRescoringEnabled() {
        int dimension = 16;
        float[] queryVector = new float[] {
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());
        addKNNDocs(INDEX_NAME, FIELD_NAME, dimension, 0, 5);

        // Search with default rescore
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", 5)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        validateExplainSearchResponse(
            queryBuilder,
            DISK_BASED_SEARCH,
            ANN_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            "shard level rescoring enabled",
            String.valueOf(dimension)
        );
    }

    @SneakyThrows
    public void testExplain_whenDiskBasedSearchRescoringDisabled_thenSucceed() {
        int dimension = 16;
        float[] queryVector = new float[] {
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f,
            1.0f,
            2.0f };
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());
        addKNNDocs(INDEX_NAME, FIELD_NAME, dimension, 0, 5);

        // Search without rescore
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", 5)
            .field(RescoreParser.RESCORE_PARAMETER, false)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        validateExplainSearchResponse(
            queryBuilder,
            DISK_BASED_SEARCH,
            ANN_SEARCH,
            VectorDataType.FLOAT.name(),
            SpaceType.L2.getValue(),
            "shard level rescoring disabled",
            String.valueOf(dimension)
        );
    }

    private void validateExplainSearchResponse(XContentBuilder queryBuilder, String... descriptions) throws IOException, ParseException {
        String responseBody = EntityUtils.toString(performSearch(INDEX_NAME, queryBuilder.toString(), "explain=true").getEntity());
        List<Object> searchResponseHits = parseSearchResponseHits(responseBody);
        searchResponseHits.stream().forEach(hit -> {
            Map<String, Object> hitMap = (Map<String, Object>) hit;
            Double score = (Double) hitMap.get("_score");
            String explanation = hitMap.get("_explanation").toString();
            assertNotNull(explanation);
            for (String description : descriptions) {
                assertTrue(explanation.contains(description));
            }
            assertTrue(explanation.contains(String.valueOf(score)));
        });
    }

    private void createDefaultKnnIndex(int dimension) throws IOException {
        // Create Mappings
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        final String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);
    }

    private void setupKNNIndexForFilterQuery() throws Exception {
        createDefaultKnnIndex(3);
        addKnnDocWithAttributes("doc1", new float[] { 6.0f, 7.9f, 3.1f }, ImmutableMap.of("color", "red", "taste", "sweet"));
        addKnnDocWithAttributes("doc2", new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of("color", "green"));
        addKnnDocWithAttributes("doc3", new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of("color", "red"));

        refreshIndex(INDEX_NAME);
    }

    private void indexTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] indexVector = new float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDocWithAttributes(indexName, Integer.toString(i), fieldName, indexVector, ImmutableMap.of("rating", String.valueOf(i)));
        }

        // Assert that all docs are ingested
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }
}
