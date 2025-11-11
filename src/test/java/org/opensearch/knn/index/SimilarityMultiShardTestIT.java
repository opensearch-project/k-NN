/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNCommonSettingsBuilder;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class SimilarityMultiShardTestIT extends KNNSingleNodeTestCase {

    private final String spaceType;
    private final String engine;
    private final String groundTruthFile;
    private final Settings settings;

    public SimilarityMultiShardTestIT(String spaceType, String engine, String groundTruthFile, Settings settings) {
        this.spaceType = spaceType;
        this.engine = engine;
        this.groundTruthFile = groundTruthFile;
        this.settings = settings;
    }

    @ParametersFactory
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            new Object[][] {
                {
                    SpaceType.COSINESIMIL.getValue(),
                    KNNEngine.FAISS.getName(),
                    "data/test_sanity_ground_truth_cosine_1000.json",
                    KNNCommonSettingsBuilder.defaultSettings().multiShard().build() },
                {
                    SpaceType.COSINESIMIL.getValue(),
                    KNNEngine.FAISS.getName(),
                    "data/test_sanity_ground_truth_cosine_1000.json",
                    KNNCommonSettingsBuilder.defaultSettings().memOptSearch().multiShard().build() } }
        );
    }

    public void testSimilaritySearchMultiShard() throws IOException, InterruptedException, ExecutionException {
        String indexName = "similarity-test-index";
        String fieldName = "vector";
        int dimensions = 16;

        createIndex(indexName, this.settings);
        createMapping(indexName, fieldName, dimensions);

        List<Object> groundTruth = loadGroundTruth(groundTruthFile);
        float[] queryVector = setupTestFromGroundTruth(indexName, fieldName, dimensions, groundTruth);

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(fieldName).vector(queryVector).k(10).build();

        SearchResponse response = client().prepareSearch(indexName).setQuery(knnQueryBuilder).get();

        validateResults(response, groundTruth);
    }

    private void createMapping(String indexName, String fieldName, int dimensions) throws IOException {
        PutMappingRequest request = new PutMappingRequest(indexName);
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().startObject("properties");
        xContentBuilder.startObject(fieldName);
        xContentBuilder.field("type", "knn_vector").field("dimension", Integer.toString(dimensions));
        xContentBuilder.startObject(KNN_METHOD);
        xContentBuilder.field(NAME, METHOD_HNSW);
        xContentBuilder.field(KNN_ENGINE, engine);
        xContentBuilder.field(METHOD_PARAMETER_SPACE_TYPE, spaceType);
        xContentBuilder.endObject();
        xContentBuilder.endObject();
        xContentBuilder.endObject();
        xContentBuilder.endObject();
        request.source(xContentBuilder);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    private void validateResults(SearchResponse response, List<Object> groundTruth) {
        assertEquals(10, response.getHits().getHits().length);
        for (int i = 0; i < 10; i++) {
            Map<String, Object> expectedDoc = (Map<String, Object>) groundTruth.get(i);
            String expectedId = (String) expectedDoc.get("id");
            float expectedScore = ((Number) expectedDoc.get("score")).floatValue();
            assertEquals(expectedId, response.getHits().getAt(i).getId());
            assertEquals(expectedScore, response.getHits().getAt(i).getScore(), 0.001f);
        }
    }
}
