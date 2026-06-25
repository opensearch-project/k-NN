/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

public class MOSFaissFloatIndexIT extends AbstractMemoryOptimizedKnnSearchIT {

    @ExpectRemoteBuildValidation
    public void testNonNestedFloatIndexWithL2() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedFloatIndexWithL2() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.L2, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedFloatIndexWithIP() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedFloatIndexWithIP() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.INNER_PRODUCT, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNonNestedFloatIndexWithCosine() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    @ExpectRemoteBuildValidation
    public void testNestedFloatIndexWithCosine() {
        doTestNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, SpaceType.COSINESIMIL, NO_ADDITIONAL_SETTINGS);
    }

    public void testWhenNoIndexBuilt() {
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, true, SpaceType.L2, NO_BUILD_HNSW);
        doTestNonNestedIndex(VectorDataType.FLOAT, EMPTY_PARAMS, false, SpaceType.L2, NO_BUILD_HNSW);
    }

    /**
     * Test that when Lucene's HNSW exhausts its visit budget due to a low-selectivity filter,
     * the MOS path falls back to exact search. We verify this by checking that the explain output
     * contains the exhausted budget message.
     */
    @SneakyThrows
    public void testFilteredSearch_whenVisitBudgetExhausted_thenFallbackToExactSearch() {
        String indexName = INDEX_NAME + "_budget_exhausted";
        String filterFieldName = "rarity";
        int dimension = 32;
        int k = 5;
        int totalDocs = 200;
        int matchingDocs = 10;

        // Create MOS-enabled index with HNSW
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "faiss")
            .field("space_type", "l2")
            .startObject("parameters")
            .field("m", 16)
            .field("ef_construction", 100)
            .field("ef_search", 1)
            .endObject()
            .endObject()
            .endObject()
            .startObject(filterFieldName)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
            .build();
        createKnnIndex(indexName, settings, mapping);

        // Index docs: only matchingDocs have "rare", rest have "common"
        for (int i = 0; i < totalDocs; i++) {
            float[] vector = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                vector[d] = (float) (i * 7 + d * 3) % 100;
            }
            String filterValue = i < matchingDocs ? "rare" : "common";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, vector, ImmutableMap.of(filterFieldName, filterValue));
        }

        refreshIndex(indexName);
        forceMergeKnnIndex(indexName);

        // Disable pre-ANN exact search threshold
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 0));

        // Query near doc 0
        float[] queryVector = new float[dimension];
        for (int d = 0; d < dimension; d++) {
            queryVector[d] = (float) (d * 3) % 100;
        }

        // Build query with filter and minimal ef_search
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(k)
            .filter(QueryBuilders.termQuery(filterFieldName, "rare"))
            .methodParameters(Map.of(METHOD_PARAMETER_EF_SEARCH, 1))
            .build();
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(queryBuilder, ToXContent.EMPTY_PARAMS);
        queryBuilder.endObject().endObject();

        Response response = performSearch(indexName, queryBuilder.toString(), "explain=true");
        String entity = EntityUtils.toString(response.getEntity());
        List<Object> hits = parseSearchResponseHits(entity);

        // We should get k results
        assertEquals(k, hits.size());

        // Verify explain confirms the exhausted budget triggered exact search
        for (Object hit : hits) {
            Map<String, Object> hitMap = (Map<String, Object>) hit;
            String explanation = hitMap.get("_explanation").toString();
            assertTrue(
                "Expected explanation to mention exhausted search budget, got: " + explanation,
                explanation.contains("since lucene vector search has exhausted number of steps")
            );
        }

        deleteKNNIndex(indexName);
    }
}
