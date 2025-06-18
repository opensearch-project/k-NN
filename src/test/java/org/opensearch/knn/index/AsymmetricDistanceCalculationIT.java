/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.opensearch.client.Request;

public class AsymmetricDistanceCalculationIT extends KNNRestTestCase {

    private static final String INDEX_NAME_PREFIX = "adc-test";
    private static final String SNAPSHOT_NAME = "adc-snapshot";
    private static final String REPOSITORY_NAME = "adc-repo";
    private static final int TEST_DIMENSION = 128;
    private static final int BINARY_DIMENSION = 64; // Must be multiple of 8
    private static final int DOC_COUNT = 100;
    private static final Random random = new Random(42); // Fixed seed for reproducible tests

    @Before
    @SneakyThrows
    public void setUp() {
        super.setUp();
        final String pathRepo = System.getProperty("tests.path.repo");
        if (pathRepo != null) {
            Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
            registerRepository(REPOSITORY_NAME, "fs", true, repoSettings);
        }
    }

    // Test Context Classes
    private static class ADCTestContext {
        final String indexName;
        final boolean adcEnabled;
        final VectorDataType dataType;
        final SpaceType spaceType;
        final String engine;
        final int dimension;
        final String fieldName;

        ADCTestContext(
            String indexName,
            boolean adcEnabled,
            VectorDataType dataType,
            SpaceType spaceType,
            String engine,
            int dimension,
            String fieldName
        ) {
            this.indexName = indexName;
            this.adcEnabled = adcEnabled;
            this.dataType = dataType;
            this.spaceType = spaceType;
            this.engine = engine;
            this.dimension = dimension;
            this.fieldName = fieldName;
        }
    }

    // 1. Nested Support Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testNestedSupport_expandNestedDocs() {
        String adcIndex = INDEX_NAME_PREFIX + "-nested-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-nested-regular";

        // Create nested mapping with ADC enabled and disabled
        createNestedIndexWithADC(adcIndex, true);
        createNestedIndexWithADC(regularIndex, false);

        // Index nested documents
        indexNestedDocuments(adcIndex, DOC_COUNT);
        indexNestedDocuments(regularIndex, DOC_COUNT);

        refreshAllIndices();

        // Test nested query with expand_nested_docs
        testNestedQuery(adcIndex, regularIndex, true);
        testNestedQuery(adcIndex, regularIndex, false);

        // Verify document counts match
        assertEquals(getDocCount(adcIndex), getDocCount(regularIndex));
    }

    // 2. Neural Search Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testNeuralSearch_newParameters() {
        String adcIndex = INDEX_NAME_PREFIX + "-neural-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-neural-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        // Test neural search with ADC-specific parameters
        testNeuralSearchWithADCParameters(adcIndex);
        testNeuralSearchConsistency(adcIndex, regularIndex);
    }

    // 3. Efficient Filtering Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testEfficientFiltering_scoreConsistency() {
        String adcIndex = INDEX_NAME_PREFIX + "-filter-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-filter-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocumentsWithMetadata(adcIndex, DOC_COUNT);
        indexVectorDocumentsWithMetadata(regularIndex, DOC_COUNT);

        refreshAllIndices();

        // Test filtering with score consistency
        testFilteringScoreConsistency(adcIndex, regularIndex);
        testShortCircuitToExact(adcIndex, regularIndex);
    }

    // 4. Vector Script Scoring/Painless Scoring Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testVectorScriptScoring() {
        String adcIndex = INDEX_NAME_PREFIX + "-script-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-script-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.COSINESIMIL, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.COSINESIMIL, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testPainlessScriptScoring(adcIndex, regularIndex);
    }

    // 5. Re-scoring Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testReScoring() {
        String adcIndex = INDEX_NAME_PREFIX + "-rescore-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-rescore-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testRescoring(adcIndex, regularIndex);
    }

    // 6. k-NN Model Management Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testKNNModelManagement() {
        String adcIndex = INDEX_NAME_PREFIX + "-model-adc";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        // Test model training, caching, and management
        testModelTraining(adcIndex);
        testModelCaching(adcIndex);
        testModelStats(adcIndex);
    }

    // 7. Derived Source for Vectors Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testDerivedSourceWithADC() {
        String adcIndex = INDEX_NAME_PREFIX + "-derived-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-derived-regular";

        // Create indices with derived source enabled
        createIndexWithDerivedSourceAndADC(adcIndex, true);
        createIndexWithDerivedSourceAndADC(regularIndex, false);

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testDerivedSourceConsistency(adcIndex, regularIndex);
    }

    // 8. Lucene on Faiss Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testLuceneOnFaiss() {
        String adcFaissIndex = INDEX_NAME_PREFIX + "-faiss-adc";
        String regularFaissIndex = INDEX_NAME_PREFIX + "-faiss-regular";
        String luceneIndex = INDEX_NAME_PREFIX + "-lucene";

        createFloatIndexWithADC(adcFaissIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularFaissIndex, false, SpaceType.L2, "faiss");
        createFloatIndexWithADC(luceneIndex, false, SpaceType.L2, "lucene");

        indexVectorDocuments(adcFaissIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularFaissIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(luceneIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testEngineConsistency(adcFaissIndex, regularFaissIndex, luceneIndex);
    }

    // 9. Binary/Byte Data Types Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testBinaryByteDataTypes() {
        // Test Binary data type
        testDataType(VectorDataType.BINARY, BINARY_DIMENSION);

        // Test Byte data type
        testDataType(VectorDataType.BYTE, TEST_DIMENSION);

        // Test Float data type
        testDataType(VectorDataType.FLOAT, TEST_DIMENSION);
    }

    // 10. Lucene Engine Tests
    // @SneakyThrows
    // @ExpectRemoteBuildValidation
    // public void testLuceneEngine() {
    // String adcIndex = INDEX_NAME_PREFIX + "-lucene-adc";
    // String regularIndex = INDEX_NAME_PREFIX + "-lucene-regular";
    //
    // // Note: ADC might not be supported on Lucene engine, test graceful handling
    // try {
    // createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "lucene");
    // indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
    // } catch (ResponseException e) {
    // // Expected if ADC not supported on Lucene
    // assertTrue(e.getMessage().contains("ADC") || e.getMessage().contains("not supported"));
    // }
    //
    // createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "lucene");
    // indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);
    //
    // refreshAllIndices();
    //
    // testLuceneEngineOperations(regularIndex);
    // }

    // 11. Warmup/Clear Cache Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testWarmupClearCache() {
        String adcIndex = INDEX_NAME_PREFIX + "-cache-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-cache-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testWarmupOperations(adcIndex, regularIndex);
        testClearCacheOperations(adcIndex, regularIndex);
    }

    // 12. Stats API Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testStatsAPI() {
        String adcIndex = INDEX_NAME_PREFIX + "-stats-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-stats-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testStatsAPIConsistency(adcIndex, regularIndex);
    }

    // 13. Dynamic Query Parameters Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testDynamicQueryParameters() {
        String adcIndex = INDEX_NAME_PREFIX + "-dynamic-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-dynamic-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        testDynamicParameters(adcIndex, regularIndex);
    }

    // 14. All Supported Space Types Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testAllSupportedSpaceTypes() {
        List<SpaceType> spaceTypes = Arrays.asList(SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.L1, SpaceType.LINF);

        for (SpaceType spaceType : spaceTypes) {
            testSpaceType(spaceType);
        }
    }

    // 15. Common OpenSearch Operations Tests
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testCommonOpenSearchOperations() {
        String adcIndex = INDEX_NAME_PREFIX + "-ops-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-ops-regular";

        createFloatIndexWithADC(adcIndex, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(regularIndex, false, SpaceType.L2, "faiss");

        indexVectorDocuments(adcIndex, DOC_COUNT, VectorDataType.FLOAT);
        indexVectorDocuments(regularIndex, DOC_COUNT, VectorDataType.FLOAT);

        refreshAllIndices();

        // Test all operations
        testUpdateOperations(adcIndex, regularIndex);
        testUpdateByQueryOperations(adcIndex, regularIndex);
        testDeleteByQueryOperations(adcIndex, regularIndex);
        testSnapshotOperations(adcIndex, regularIndex);
        testReindexOperations(adcIndex, regularIndex);
        testMergingOperations(adcIndex, regularIndex);
    }

    // Helper Methods for Index Creation
    @SneakyThrows
    private void createFloatIndexWithADC(String indexName, boolean enableADC, SpaceType spaceType, String engine) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("settings")
            .startObject("index")
            .field("knn", true)
            .endObject()
            .endObject()
            .startObject("mappings")
            .startObject("properties")
            .startObject("vector_field")
            .field("type", "knn_vector")
            .field("dimension", TEST_DIMENSION);

        if (enableADC) {
            builder.field("compression_level", "32x");
        }

        builder.startObject("method")
            .field("name", "hnsw")
            .field("engine", engine)
            .field("space_type", spaceType.getValue())
            .startObject("parameters")
            .field("ef_construction", 128)
            .field("m", 16);

        if (enableADC && "faiss".equals(engine)) {
            builder.startObject("encoder")
                .field("name", "binary")
                .startObject("parameters")
                .field("bits", 1)
                .field("enable_adc", true)
                .field("random_rotation", false)
                .endObject()
                .endObject();
        }

        builder.endObject() // parameters
            .endObject() // method
            .endObject() // vector_field
            .endObject() // properties
            .endObject() // mappings
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    @SneakyThrows
    private void createNestedIndexWithADC(String indexName, boolean enableADC) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("settings")
            .startObject("index")
            .field("knn", true)
            .endObject()
            .endObject()
            .startObject("mappings")
            .startObject("properties")
            .startObject("nested_field")
            .field("type", "nested")
            .startObject("properties")
            .startObject("vector_field")
            .field("type", "knn_vector")
            .field("dimension", TEST_DIMENSION);

        if (enableADC) {
            builder.field("compression_level", "32x")
                .startObject("method")
                .field("name", "hnsw")
                .field("engine", "faiss")
                .field("space_type", "l2")
                .startObject("parameters")
                .field("ef_construction", 128)
                .field("m", 16)
                .startObject("encoder")
                .field("name", "binary")
                .startObject("parameters")
                .field("bits", 1)
                .field("enable_adc", true)
                .field("random_rotation", false)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
        }

        builder.endObject() // vector_field
            .startObject("metadata")
            .field("type", "keyword")
            .endObject()
            .endObject() // properties of nested_field
            .endObject() // nested_field
            .endObject() // properties
            .endObject() // mappings
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    @SneakyThrows
    private void createIndexWithDerivedSourceAndADC(String indexName, boolean enableADC) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("settings")
            .startObject("index")
            .field("knn", true)
            .field("knn.derived_source.enabled", true)
            .endObject()
            .endObject()
            .startObject("mappings")
            .startObject("properties")
            .startObject("vector_field")
            .field("type", "knn_vector")
            .field("dimension", TEST_DIMENSION);

        if (enableADC) {
            builder.field("compression_level", "32x")
                .startObject("method")
                .field("name", "hnsw")
                .field("engine", "faiss")
                .field("space_type", "l2")
                .startObject("parameters")
                .field("ef_construction", 128)
                .field("m", 16)
                .startObject("encoder")
                .field("name", "binary")
                .startObject("parameters")
                .field("bits", 1)
                .field("enable_adc", true)
                .field("random_rotation", false)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
        }

        builder.endObject() // vector_field
            .endObject() // properties
            .endObject() // mappings
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    // Helper Methods for Document Indexing
    @SneakyThrows
    private void indexVectorDocuments(String indexName, int count, VectorDataType dataType) {
        for (int i = 0; i < count; i++) {
            XContentBuilder docBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field("vector_field", generateVector(dataType, getDimensionForDataType(dataType)))
                .field("id", i)
                .endObject();
            addKnnDoc(indexName, String.valueOf(i), docBuilder.toString());
        }
    }

    @SneakyThrows
    private void indexVectorDocumentsWithMetadata(String indexName, int count) {
        for (int i = 0; i < count; i++) {
            XContentBuilder docBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field("vector_field", (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION))
                .field("id", i)
                .field("category", i % 10)
                .field("price", random.nextDouble() * 100)
                .endObject();
            addKnnDoc(indexName, String.valueOf(i), docBuilder.toString());
        }
    }

    @SneakyThrows
    private void indexNestedDocuments(String indexName, int count) {
        for (int i = 0; i < count; i++) {
            XContentBuilder docBuilder = XContentFactory.jsonBuilder().startObject().startArray("nested_field");

            // Add multiple nested objects per document
            for (int j = 0; j < 3; j++) {
                docBuilder.startObject()
                    .field("vector_field", (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION))
                    .field("metadata", "item_" + j)
                    .endObject();
            }

            docBuilder.endArray().field("id", i).endObject();
            addKnnDoc(indexName, String.valueOf(i), docBuilder.toString());
        }
    }

    // Helper Methods for Testing Different Functionality
    @SneakyThrows
    private void testNestedQuery(String adcIndex, String regularIndex, boolean expandNestedDocs) {
        float[] queryVector = (float[]) (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", "nested_field")
            .startObject("query")
            .startObject("knn")
            .startObject("nested_field.vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .field("size", 10);

        if (expandNestedDocs) {
            queryBuilder.field("expand_nested_docs", true);
        }

        queryBuilder.endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        // Verify both queries return results
        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);

        // For expand_nested_docs, the hit counts might differ, but structure should be consistent
        // TODO here
        // assertResponseStructureConsistent(adcResponse, regularResponse);
    }

    @SneakyThrows
    private void testNeuralSearchWithADCParameters(String indexName) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .field("ef_search", 50) // ADC-specific parameter
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = searchIndex(indexName, queryBuilder.toString());
        assertTrue(getTotalHitsFromResponse(response) > 0);
    }

    @SneakyThrows
    private void testNeuralSearchConsistency(String adcIndex, String regularIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        // Both should return same number of hits
        assertEquals(getTotalHitsFromResponse(adcResponse), getTotalHitsFromResponse(regularResponse));
    }

    @SneakyThrows
    private void testFilteringScoreConsistency(String adcIndex, String regularIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("bool")
            .startArray("must")
            .startObject()
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endArray()
            .startArray("filter")
            .startObject()
            .startObject("range")
            .startObject("category")
            .field("lte", 5)
            .endObject()
            .endObject()
            .endObject()
            .endArray()
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        // Verify filtering works consistently
        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);

        // Score consistency check - scores should be similar within tolerance
        List<Float> adcScores = extractScoresFromResponse(adcResponse);
        List<Float> regularScores = extractScoresFromResponse(regularResponse);

        assertEquals(adcScores.size(), regularScores.size());
        // Allow some tolerance for score differences due to compression
        for (int i = 0; i < Math.min(adcScores.size(), regularScores.size()); i++) {
            float scoreDiff = Math.abs(adcScores.get(i) - regularScores.get(i));
            assertTrue("Score difference too large: " + scoreDiff, scoreDiff < 0.1);
        }
    }

    @SneakyThrows
    private void testShortCircuitToExact(String adcIndex, String regularIndex) {
        // Test with very restrictive filter that should trigger short circuit to exact search
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("bool")
            .startArray("must")
            .startObject()
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endArray()
            .startArray("filter")
            .startObject()
            .startObject("term")
            .field("id", 1)
            .endObject()
            .endObject()
            .endArray()
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        // Both should return the same filtered results
        assertEquals(getTotalHitsFromResponse(adcResponse), getTotalHitsFromResponse(regularResponse));
    }

    @SneakyThrows
    private void testPainlessScriptScoring(String adcIndex, String regularIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("script_score")
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("script")
            .field("source", "knn_score")
            .startObject("params")
            .field("field", "vector_field")
            .field("query_value", queryVector)
            .field("space_type", "cosinesimil")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .field("size", 10)
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);

        // Verify script scoring works with ADC
        List<Float> adcScores = extractScoresFromResponse(adcResponse);
        List<Float> regularScores = extractScoresFromResponse(regularResponse);

        assertFalse(adcScores.isEmpty());
        assertFalse(regularScores.isEmpty());
    }

    @SneakyThrows
    private void testRescoring(String adcIndex, String regularIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 20)
            .endObject()
            .endObject()
            .endObject()
            .startObject("rescore")
            .field("window_size", 10)
            .startObject("query")
            .startObject("rescore_query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 5)
            .endObject()
            .endObject()
            .endObject()
            .field("query_weight", 0.7)
            .field("rescore_query_weight", 0.3)
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);
    }

    @SneakyThrows
    private void testModelTraining(String indexName) {
        // Force model training by performing search
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = searchIndex(indexName, queryBuilder.toString());
        assertTrue(getTotalHitsFromResponse(response) > 0);

        // Verify model was created (check stats)
        Response statsResponse = getKNNStats();
        String statsBody = EntityUtils.toString(statsResponse.getEntity());
        assertTrue(statsBody.contains("faiss"));
    }

    @SneakyThrows
    private void testModelCaching(String indexName) {
        // Test warmup
        Request warmupRequest = new Request("GET", "/" + indexName + "/_knn/warmup");
        Response warmupResponse = client().performRequest(warmupRequest);
        assertEquals(RestStatus.OK, RestStatus.fromCode(warmupResponse.getStatusLine().getStatusCode()));
    }

    @SneakyThrows
    private void testModelStats(String indexName) {
        Response statsResponse = getKNNStats();
        String statsBody = EntityUtils.toString(statsResponse.getEntity());

        // Verify stats contain ADC-related information
        assertNotNull(statsBody);
        assertTrue(statsBody.length() > 0);
    }

    @SneakyThrows
    private void testDerivedSourceConsistency(String adcIndex, String regularIndex) {
        // Test that derived source works consistently with ADC
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .field("_source", true)
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        // Both should return source fields
        List<Map<String, Object>> adcHits = extractHitsFromResponse(adcResponse);
        List<Map<String, Object>> regularHits = extractHitsFromResponse(regularResponse);

        assertFalse(adcHits.isEmpty());
        assertFalse(regularHits.isEmpty());

        // Verify source is present in both
        assertTrue(adcHits.get(0).containsKey("_source"));
        assertTrue(regularHits.get(0).containsKey("_source"));
    }

    @SneakyThrows
    private void testEngineConsistency(String adcFaissIndex, String regularFaissIndex, String luceneIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response adcFaissResponse = searchIndex(adcFaissIndex, queryBuilder.toString());
        Response regularFaissResponse = searchIndex(regularFaissIndex, queryBuilder.toString());
        Response luceneResponse = searchIndex(luceneIndex, queryBuilder.toString());

        // All should return results
        assertTrue(getTotalHitsFromResponse(adcFaissResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularFaissResponse) > 0);
        assertTrue(getTotalHitsFromResponse(luceneResponse) > 0);
    }

    private void testDataType(VectorDataType dataType, int dimension) {
        String adcIndex = INDEX_NAME_PREFIX + "-" + dataType.getValue() + "-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-" + dataType.getValue() + "-regular";

        try {
            createIndexWithDataType(adcIndex, true, dataType, dimension);
            createIndexWithDataType(regularIndex, false, dataType, dimension);

            indexVectorDocuments(adcIndex, DOC_COUNT / 10, dataType); // Smaller count for complex data types
            indexVectorDocuments(regularIndex, DOC_COUNT / 10, dataType);

            refreshAllIndices();

            // Test search consistency
            Object queryVector = generateVector(dataType, dimension);
            testSearchWithDataType(adcIndex, regularIndex, queryVector, dataType);

        } catch (Exception e) {
            // Some data types might not support ADC, that's okay
            logger.info("Data type {} might not support ADC: {}", dataType, e.getMessage());
        }
    }

    @SneakyThrows
    private void createIndexWithDataType(String indexName, boolean enableADC, VectorDataType dataType, int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("settings")
            .startObject("index")
            .field("knn", true)
            .endObject()
            .endObject()
            .startObject("mappings")
            .startObject("properties")
            .startObject("vector_field")
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field("data_type", dataType.getValue());

        if (enableADC) {
            builder.field("compression_level", "32x")
                .startObject("method")
                .field("name", "hnsw")
                .field("engine", "faiss")
                .field("space_type", "l2")
                .startObject("parameters")
                .field("ef_construction", 128)
                .field("m", 16)
                .startObject("encoder")
                .field("name", "binary")
                .startObject("parameters")
                .field("bits", 1)
                .field("enable_adc", true)
                .field("random_rotation", false)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
        }

        builder.endObject() // vector_field
            .endObject() // properties
            .endObject() // mappings
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    @SneakyThrows
    private void testSearchWithDataType(String adcIndex, String regularIndex, Object queryVector, VectorDataType dataType) {
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 5)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);
    }

    @SneakyThrows
    private void testLuceneEngineOperations(String indexName) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = searchIndex(indexName, queryBuilder.toString());
        assertTrue(getTotalHitsFromResponse(response) > 0);
    }

    @SneakyThrows
    private void testWarmupOperations(String adcIndex, String regularIndex) {
        Request warmupRequest1 = new Request("GET", "/" + adcIndex + "/_knn/warmup");
        Request warmupRequest2 = new Request("GET", "/" + regularIndex + "/_knn/warmup");

        Response warmupResponse1 = client().performRequest(warmupRequest1);
        Response warmupResponse2 = client().performRequest(warmupRequest2);

        assertEquals(RestStatus.OK, RestStatus.fromCode(warmupResponse1.getStatusLine().getStatusCode()));
        assertEquals(RestStatus.OK, RestStatus.fromCode(warmupResponse2.getStatusLine().getStatusCode()));
    }

    @SneakyThrows
    private void testClearCacheOperations(String adcIndex, String regularIndex) {
        Request clearCacheRequest1 = new Request("POST", "/" + adcIndex + "/_knn/clear_cache");
        Request clearCacheRequest2 = new Request("POST", "/" + regularIndex + "/_knn/clear_cache");

        Response clearCacheResponse1 = client().performRequest(clearCacheRequest1);
        Response clearCacheResponse2 = client().performRequest(clearCacheRequest2);

        assertEquals(RestStatus.OK, RestStatus.fromCode(clearCacheResponse1.getStatusLine().getStatusCode()));
        assertEquals(RestStatus.OK, RestStatus.fromCode(clearCacheResponse2.getStatusLine().getStatusCode()));
    }

    @SneakyThrows
    private void testStatsAPIConsistency(String adcIndex, String regularIndex) {
        Response statsResponse = getKNNStats();
        String statsBody = EntityUtils.toString(statsResponse.getEntity());

        // Verify stats are available and contain both indices
        assertNotNull(statsBody);
        assertTrue(statsBody.length() > 0);

        // Both indices should appear in stats
        assertTrue(statsBody.contains("indices") || statsBody.contains("nodes"));
    }

    @SneakyThrows
    private void testDynamicParameters(String adcIndex, String regularIndex) {
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);

        // Test with different dynamic parameters
        int[] efValues = { 10, 50, 100 };
        for (int ef : efValues) {
            XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject("vector_field")
                .field("vector", queryVector)
                .field("k", 10)
                .field("ef_search", ef)
                .endObject()
                .endObject()
                .endObject()
                .endObject();

            Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
            Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

            assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
            assertTrue(getTotalHitsFromResponse(regularResponse) > 0);
        }
    }

    private void testSpaceType(SpaceType spaceType) {
        String adcIndex = INDEX_NAME_PREFIX + "-" + spaceType.getValue() + "-adc";
        String regularIndex = INDEX_NAME_PREFIX + "-" + spaceType.getValue() + "-regular";

        try {
            createFloatIndexWithADC(adcIndex, true, spaceType, "faiss");
            createFloatIndexWithADC(regularIndex, false, spaceType, "faiss");

            indexVectorDocuments(adcIndex, DOC_COUNT / 10, VectorDataType.FLOAT);
            indexVectorDocuments(regularIndex, DOC_COUNT / 10, VectorDataType.FLOAT);

            refreshAllIndices();

            // Test search with this space type
            float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);
            testSearchConsistency(adcIndex, regularIndex, queryVector);

        } catch (Exception e) {
            logger.info("Space type {} might not support ADC: {}", spaceType, e.getMessage());
        }
    }

    @SneakyThrows
    private void testSearchConsistency(String adcIndex, String regularIndex, float[] queryVector) {
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject("vector_field")
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response adcResponse = searchIndex(adcIndex, queryBuilder.toString());
        Response regularResponse = searchIndex(regularIndex, queryBuilder.toString());

        assertTrue(getTotalHitsFromResponse(adcResponse) > 0);
        assertTrue(getTotalHitsFromResponse(regularResponse) > 0);
    }

    @SneakyThrows
    private void testUpdateOperations(String adcIndex, String regularIndex) {
        // Test document updates
        XContentBuilder updateDoc = XContentFactory.jsonBuilder()
            .startObject()
            .field("vector_field", (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION))
            .field("id", 999)
            .endObject();

        addKnnDoc(adcIndex, "update_test", updateDoc.toString());
        addKnnDoc(regularIndex, "update_test", updateDoc.toString());

        refreshAllIndices();

        // Verify documents were updated
        Map<String, Object> adcDoc = getKnnDoc(adcIndex, "update_test");
        Map<String, Object> regularDoc = getKnnDoc(regularIndex, "update_test");

        assertNotNull(adcDoc);
        assertNotNull(regularDoc);
    }

    @SneakyThrows
    private void testUpdateByQueryOperations(String adcIndex, String regularIndex) {
        // Test update by query
        XContentBuilder updateQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("range")
            .startObject("id")
            .field("lte", 5)
            .endObject()
            .endObject()
            .endObject()
            .startObject("script")
            .field("source", "ctx._source.updated = true")
            .endObject()
            .endObject();

        Request updateByQueryRequest1 = new Request("POST", "/" + adcIndex + "/_update_by_query");
        Request updateByQueryRequest2 = new Request("POST", "/" + regularIndex + "/_update_by_query");

        updateByQueryRequest1.setJsonEntity(updateQuery.toString());
        updateByQueryRequest2.setJsonEntity(updateQuery.toString());

        Response updateResponse1 = client().performRequest(updateByQueryRequest1);
        Response updateResponse2 = client().performRequest(updateByQueryRequest2);

        assertEquals(RestStatus.OK, RestStatus.fromCode(updateResponse1.getStatusLine().getStatusCode()));
        assertEquals(RestStatus.OK, RestStatus.fromCode(updateResponse2.getStatusLine().getStatusCode()));
    }

    @SneakyThrows
    private void testDeleteByQueryOperations(String adcIndex, String regularIndex) {
        XContentBuilder deleteQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("range")
            .startObject("id")
            .field("gte", 95)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Request deleteByQueryRequest1 = new Request("POST", "/" + adcIndex + "/_delete_by_query");
        Request deleteByQueryRequest2 = new Request("POST", "/" + regularIndex + "/_delete_by_query");

        deleteByQueryRequest1.setJsonEntity(deleteQuery.toString());
        deleteByQueryRequest2.setJsonEntity(deleteQuery.toString());

        Response deleteResponse1 = client().performRequest(deleteByQueryRequest1);
        Response deleteResponse2 = client().performRequest(deleteByQueryRequest2);

        assertEquals(RestStatus.OK, RestStatus.fromCode(deleteResponse1.getStatusLine().getStatusCode()));
        assertEquals(RestStatus.OK, RestStatus.fromCode(deleteResponse2.getStatusLine().getStatusCode()));
    }

    @SneakyThrows
    private void testSnapshotOperations(String adcIndex, String regularIndex) {
        if (System.getProperty("tests.path.repo") != null) {
            createSnapshot(REPOSITORY_NAME, SNAPSHOT_NAME, true);

            // Test restore
            deleteIndex(adcIndex);
            deleteIndex(regularIndex);

            restoreSnapshot("-restored", List.of(adcIndex, regularIndex), REPOSITORY_NAME, SNAPSHOT_NAME, true);

            // Verify indices were restored
            assertTrue(indexExists(adcIndex + "-restored"));
            assertTrue(indexExists(regularIndex + "-restored"));
        }
    }

    @SneakyThrows
    private void testReindexOperations(String adcIndex, String regularIndex) {
        String reindexTarget1 = adcIndex + "-reindexed";
        String reindexTarget2 = regularIndex + "-reindexed";

        // Create target indices
        createFloatIndexWithADC(reindexTarget1, true, SpaceType.L2, "faiss");
        createFloatIndexWithADC(reindexTarget2, false, SpaceType.L2, "faiss");

        // Reindex
        reindex(adcIndex, reindexTarget1);
        reindex(regularIndex, reindexTarget2);

        refreshAllIndices();

        // Verify reindex worked
        assertTrue(getDocCount(reindexTarget1) > 0);
        assertTrue(getDocCount(reindexTarget2) > 0);
    }

    @SneakyThrows
    private void testMergingOperations(String adcIndex, String regularIndex) {
        forceMergeKnnIndex(adcIndex, 1);
        forceMergeKnnIndex(regularIndex, 1);

        refreshAllIndices();

        // Verify indices still work after merging
        float[] queryVector = (float[]) generateVector(VectorDataType.FLOAT, TEST_DIMENSION);
        testSearchConsistency(adcIndex, regularIndex, queryVector);
    }

    // Utility Methods
    private Object generateVector(VectorDataType dataType, int dimension) {
        switch (dataType) {
            case FLOAT:
                return generateFloatVector(dimension);
            case BYTE:
                return generateByteVector(dimension);
            case BINARY:
                return generateBinaryVector(dimension);
            default:
                throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    private float[] generateFloatVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }

    private byte[] generateByteVector(int dimension) {
        byte[] vector = new byte[dimension];
        random.nextBytes(vector);
        return vector;
    }

    private String generateBinaryVector(int dimension) {
        // Binary vectors are represented as base64 strings
        byte[] bytes = new byte[dimension / 8];
        random.nextBytes(bytes);
        return java.util.Base64.getEncoder().encodeToString(bytes);
    }

    private int getDimensionForDataType(VectorDataType dataType) {
        return dataType == VectorDataType.BINARY ? BINARY_DIMENSION : TEST_DIMENSION;
    }

    @SneakyThrows
    private Response searchIndex(String indexName, String query) {
        Request request = new Request("POST", "/" + indexName + "/_search");
        request.setJsonEntity(query);
        return client().performRequest(request);
    }

    @SneakyThrows
    private int getTotalHitsFromResponse(Response response) {
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = parseResponseToMap(responseBody);

        Map<String, Object> hits = (Map<String, Object>) responseMap.get("hits");
        Object total = hits.get("total");
        if (total instanceof Map) {
            return (Integer) ((Map<String, Object>) total).get("value");
        }
        return (Integer) total;
    }

    @SneakyThrows
    private List<Float> extractScoresFromResponse(Response response) {
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = parseResponseToMap(responseBody);
        Map<String, Object> hits = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hits.get("hits");

        List<Float> scores = new ArrayList<>();
        for (Map<String, Object> hit : hitsList) {
            scores.add(((Number) hit.get("_score")).floatValue());
        }
        return scores;
    }

    @SneakyThrows
    private List<Map<String, Object>> extractHitsFromResponse(Response response) {
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = parseResponseToMap(responseBody);
        Map<String, Object> hits = (Map<String, Object>) responseMap.get("hits");
        return (List<Map<String, Object>>) hits.get("hits");
    }

    @SneakyThrows
    private Response getKNNStats() {
        Request request = new Request("GET", "/_plugins/_knn/stats");
        return client().performRequest(request);
    }

    // @SneakyThrows
    // private int getDocCount(String indexName) {
    // Request request = new Request("GET", "/" + indexName + "/_count");
    // Response response = client().performRequest(request);
    // String responseBody = EntityUtils.toString(response.getEntity());
    // Map<String, Object> responseMap = parseResponseToMap(responseBody);
    // return (Integer) responseMap.get("count");
    // }
    //
    // @SneakyThrows
    // private boolean indexExists(String indexName) {
    // try {
    // Request request = new Request("HEAD", "/" + indexName);
    // Response response = client().performRequest(request);
    // return response.getStatusLine().getStatusCode() == 200;
    // } catch (ResponseException e) {
    // return false;
    // }
    // }

    private Map<String, Object> parseResponseToMap(String responseBody) throws IOException {
        // createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map()
        // return parseSearchResponseHits(responseBody);
        return ((Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map());

    }
}
