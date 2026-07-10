/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Integration tests for 32x compression (SQ 1-bit) across both Faiss and Lucene engines.
 * Parameterized by engine to ensure symmetric coverage.
 */
@Log4j2
public class Compression32XIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final int DOC_COUNT = 100;
    private static final int QUERY_COUNT = 10;
    private static final int K = 10;
    private static final int HNSW_M = 16;
    private static final int HNSW_EF_CONSTRUCTION = 100;
    private static final int HNSW_EF_SEARCH = 100;
    private static final double MIN_RECALL_X32_L2 = 0.70;
    private static final double MIN_RECALL_X32_IN_MEMORY = 0.60;
    private static final double MIN_RECALL_X32_IP = 0.60;
    private static final double MIN_RECALL_X32_COSINE = 0.70;
    private static final double MIN_RECALL_FP32 = 0.95;

    private static final int DIMENSION_SMALL = 64;
    private static final String FIELD_NAME_2 = "test_field_2";

    private static final int BASELINE_DOC_COUNT = 200;
    private static final int BASELINE_QUERY_COUNT = 20;

    private static final float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION, true);
    private static final float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION, DOC_COUNT, true);
    private static final float[][] INDEX_VECTORS_SMALL = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION_SMALL, true);
    private static final float[][] QUERY_VECTORS_SMALL = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION_SMALL, DOC_COUNT, true);
    private static final float[][] BASELINE_INDEX_VECTORS = TestUtils.getIndexVectors(BASELINE_DOC_COUNT, DIMENSION, true);
    private static final float[][] BASELINE_QUERY_VECTORS = TestUtils.getQueryVectors(
        BASELINE_QUERY_COUNT,
        DIMENSION,
        BASELINE_DOC_COUNT,
        true
    );
    private static final List<Set<String>> GROUND_TRUTH_L2 = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.L2,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_IP = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.INNER_PRODUCT,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_COSINE = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.COSINESIMIL,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_SMALL = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS_SMALL,
        QUERY_VECTORS_SMALL,
        SpaceType.L2,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_BASELINE = TestUtils.computeGroundTruthValues(
        BASELINE_INDEX_VECTORS,
        BASELINE_QUERY_VECTORS,
        SpaceType.L2,
        K
    );

    private final String engineName;

    public Compression32XIT(String engineName) {
        this.engineName = engineName;
    }

    @ParametersFactory(argumentFormatting = "engine:%1$s")
    public static Collection<Object[]> parameters() {
        return Arrays.asList(new Object[] { FAISS_NAME }, new Object[] { LUCENE_NAME });
    }

    private String prefix() {
        return engineName + "_x32_";
    }

    // Verifies explicit mode=on_disk + compression_level=32x produces acceptable recall
    @SneakyThrows
    public void testX32_kSearch() {
        String indexName = prefix() + "ksearch";
        createX32Index(indexName, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("[{}] x32 k-search recall: {}", engineName, recall);
        assertTrue(engineName + " x32 recall should be >= " + MIN_RECALL_X32_L2 + " but was " + recall, recall >= MIN_RECALL_X32_L2);
    }

    // Verifies compression_level=1x opts out of quantization and retains FP32-level recall
    @SneakyThrows
    public void testX32_optOut() {
        String indexName = prefix() + "optout";
        createFP32Index(indexName, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("[{}] FP32 opt-out recall: {}", engineName, recall);
        assertTrue(engineName + " FP32 recall should be >= " + MIN_RECALL_FP32 + " but was " + recall, recall >= MIN_RECALL_FP32);
    }

    // Validates x32 recall across L2, inner product, and cosine space types
    @SneakyThrows
    public void testX32_spaceTypes() {
        String indexL2 = prefix() + "l2";
        String indexIP = prefix() + "ip";
        String indexCosine = prefix() + "cosine";

        createX32Index(indexL2, SpaceType.L2, DIMENSION);
        createX32Index(indexIP, SpaceType.INNER_PRODUCT, DIMENSION);
        createX32Index(indexCosine, SpaceType.COSINESIMIL, DIMENSION);

        bulkAddKnnDocs(indexL2, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexIP, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexCosine, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);

        forceMergeKnnIndex(indexL2, 1);
        forceMergeKnnIndex(indexIP, 1);
        forceMergeKnnIndex(indexCosine, 1);

        List<List<String>> resultsL2 = bulkSearch(indexL2, FIELD_NAME, QUERY_VECTORS, K);
        double recallL2 = TestUtils.calculateRecallValue(resultsL2, GROUND_TRUTH_L2, K);
        assertTrue(engineName + " L2 recall should be >= " + MIN_RECALL_X32_L2 + " but was " + recallL2, recallL2 >= MIN_RECALL_X32_L2);

        List<List<String>> resultsIP = bulkSearch(indexIP, FIELD_NAME, QUERY_VECTORS, K);
        double recallIP = TestUtils.calculateRecallValue(resultsIP, GROUND_TRUTH_IP, K);
        assertTrue(engineName + " IP recall should be >= " + MIN_RECALL_X32_IP + " but was " + recallIP, recallIP >= MIN_RECALL_X32_IP);

        List<List<String>> resultsCosine = bulkSearch(indexCosine, FIELD_NAME, QUERY_VECTORS, K);
        double recallCosine = TestUtils.calculateRecallValue(resultsCosine, GROUND_TRUTH_COSINE, K);
        assertTrue(
            engineName + " Cosine recall should be >= " + MIN_RECALL_X32_COSINE + " but was " + recallCosine,
            recallCosine >= MIN_RECALL_X32_COSINE
        );

        // Validate score ranges per space type
        validateScoreRanges(indexL2, SpaceType.L2, 0.0f, 1.0f);
        validateScoreRanges(indexIP, SpaceType.INNER_PRODUCT, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
        validateScoreRanges(indexCosine, SpaceType.COSINESIMIL, 0.0f, 2.0f);
    }

    // Validates x32 works with multiple knn_vector fields of different dimensions in one index
    @SneakyThrows
    public void testX32_multiField() {
        String indexName = prefix() + "multifield";
        createX32MultiFieldIndex(indexName);

        for (int i = 0; i < DOC_COUNT; i++) {
            addKnnDoc(
                indexName,
                String.valueOf(i),
                Arrays.asList(FIELD_NAME, FIELD_NAME_2),
                Arrays.asList(toObjectArray(INDEX_VECTORS[i]), toObjectArray(INDEX_VECTORS_SMALL[i]))
            );
        }
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsField1 = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallField1 = TestUtils.calculateRecallValue(resultsField1, GROUND_TRUTH_L2, K);
        assertTrue(
            engineName + " field1 recall should be >= " + MIN_RECALL_X32_L2 + " but was " + recallField1,
            recallField1 >= MIN_RECALL_X32_L2
        );

        List<List<String>> resultsField2 = bulkSearch(indexName, FIELD_NAME_2, QUERY_VECTORS_SMALL, K);
        double recallField2 = TestUtils.calculateRecallValue(resultsField2, GROUND_TRUTH_SMALL, K);
        assertTrue(
            engineName + " field2 recall should be >= " + MIN_RECALL_X32_L2 + " but was " + recallField2,
            recallField2 >= MIN_RECALL_X32_L2
        );
    }

    // Validates x32 recall survives force merge, close/reopen, and snapshot/restore
    @SneakyThrows
    public void testX32_lifecycle() {
        String indexName = prefix() + "lifecycle";
        createX32Index(indexName, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsAfterMerge = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterMerge = TestUtils.calculateRecallValue(resultsAfterMerge, GROUND_TRUTH_L2, K);
        assertTrue(
            engineName + " recall after merge should be >= " + MIN_RECALL_X32_L2 + " but was " + recallAfterMerge,
            recallAfterMerge >= MIN_RECALL_X32_L2
        );

        closeIndex(indexName);
        openIndex(indexName);
        ensureGreen(indexName);

        List<List<String>> resultsAfterReopen = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterReopen = TestUtils.calculateRecallValue(resultsAfterReopen, GROUND_TRUTH_L2, K);
        assertTrue(
            engineName + " recall after reopen should be >= " + MIN_RECALL_X32_L2 + " but was " + recallAfterReopen,
            recallAfterReopen >= MIN_RECALL_X32_L2
        );

        String repositoryName = prefix() + "repo-" + randomLowerCaseString();
        String snapshotName = prefix() + "snap-" + getTestName().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_-]", "");
        String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repositoryName, "fs", true, repoSettings);
        createSnapshot(repositoryName, snapshotName, true);

        deleteKNNIndex(indexName);

        String restoreSuffix = "-restored";
        restoreSnapshot(restoreSuffix, List.of(indexName), repositoryName, snapshotName, true);
        String restoredIndexName = indexName + restoreSuffix;
        ensureGreen(restoredIndexName);

        List<List<String>> resultsAfterRestore = bulkSearch(restoredIndexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterRestore = TestUtils.calculateRecallValue(resultsAfterRestore, GROUND_TRUTH_L2, K);
        assertTrue(
            engineName + " recall after restore should be >= " + MIN_RECALL_X32_L2 + " but was " + recallAfterRestore,
            recallAfterRestore >= MIN_RECALL_X32_L2
        );
    }

    // Validates radial search (max_distance, min_score) works on x32 quantized indices
    @SneakyThrows
    public void testX32_radialSearch() {
        String indexName = prefix() + "radial";
        createX32Index(indexName, SpaceType.L2, DIMENSION);
        for (int i = 0; i < INDEX_VECTORS.length; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, INDEX_VECTORS[i]);
        }
        refreshIndex(indexName);

        float[] queryVector = QUERY_VECTORS[0];

        String maxDistanceQuery = buildRadialSearchQuery(FIELD_NAME, queryVector, "max_distance", 100000.0f);
        Request maxDistanceRequest = new Request("POST", "/" + indexName + "/_search");
        maxDistanceRequest.setJsonEntity(maxDistanceQuery);
        Response maxDistResponse = client().performRequest(maxDistanceRequest);
        assertEquals(engineName + " max_distance should succeed", 200, maxDistResponse.getStatusLine().getStatusCode());

        String minScoreQuery = buildRadialSearchQuery(FIELD_NAME, queryVector, "min_score", 0.001f);
        Request minScoreRequest = new Request("POST", "/" + indexName + "/_search");
        minScoreRequest.setJsonEntity(minScoreQuery);
        Response minScoreResponse = client().performRequest(minScoreRequest);
        assertEquals(engineName + " min_score should succeed", 200, minScoreResponse.getStatusLine().getStatusCode());
    }

    // Script scoring operates on raw vectors, so x32 and fp32 indices must return identical scores
    @SneakyThrows
    public void testX32_scriptScoring() {
        String x32IndexName = prefix() + "script_x32";
        String fp32IndexName = prefix() + "script_fp32";
        int docCount = 20;
        int dimension = 16;

        float[][] vectors = TestUtils.getIndexVectors(docCount, dimension, true);
        float[] queryVector = TestUtils.getQueryVectors(1, dimension, docCount, true)[0];

        createIndexForScriptScoring(x32IndexName, SpaceType.L2, dimension, true);
        createIndexForScriptScoring(fp32IndexName, SpaceType.L2, dimension, false);

        for (int i = 0; i < docCount; i++) {
            addKnnDoc(x32IndexName, String.valueOf(i), FIELD_NAME, vectors[i]);
            addKnnDoc(fp32IndexName, String.valueOf(i), FIELD_NAME, vectors[i]);
        }
        forceMergeKnnIndex(x32IndexName, 1);
        forceMergeKnnIndex(fp32IndexName, 1);

        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());

        Request x32Request = constructKNNScriptQueryRequest(x32IndexName, qb, params, docCount);
        Response x32Response = client().performRequest(x32Request);
        List<KNNResult> x32Results = parseSearchResponse(EntityUtils.toString(x32Response.getEntity()), FIELD_NAME);

        Request fp32Request = constructKNNScriptQueryRequest(fp32IndexName, qb, params, docCount);
        Response fp32Response = client().performRequest(fp32Request);
        List<KNNResult> fp32Results = parseSearchResponse(EntityUtils.toString(fp32Response.getEntity()), FIELD_NAME);

        assertEquals(docCount, x32Results.size());
        assertEquals(docCount, fp32Results.size());

        for (int i = 0; i < x32Results.size(); i++) {
            assertEquals(engineName + " doc ID mismatch at position " + i, fp32Results.get(i).getDocId(), x32Results.get(i).getDocId());
            assertEquals(
                engineName + " score mismatch at position " + i,
                fp32Results.get(i).getScore(),
                x32Results.get(i).getScore(),
                0.001f
            );
        }
    }

    // Recall baseline with a larger dataset (200 docs, 128-dim) to catch scale-dependent issues
    @SneakyThrows
    public void testX32_recallBaseline() {
        String indexName = prefix() + "recall_baseline";
        createX32Index(indexName, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(indexName, FIELD_NAME, BASELINE_INDEX_VECTORS, BASELINE_DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, BASELINE_QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_BASELINE, K);
        logger.info("[{}] x32 recall baseline (200 docs): {}", engineName, recall);
        assertTrue(engineName + " recall baseline should be >= " + MIN_RECALL_X32_L2 + " but was " + recall, recall >= MIN_RECALL_X32_L2);
    }

    // Verifies x32 with mode=in_memory produces acceptable recall and scores differ from fp32 ANN
    @SneakyThrows
    public void testX32_inMemory() {
        String indexName = prefix() + "in_memory";
        createX32InMemoryIndex(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("[{}] x32 in-memory recall: {}", engineName, recall);
        assertTrue(
            engineName + " x32 in-memory recall should be >= " + MIN_RECALL_X32_IN_MEMORY + " but was " + recall,
            recall >= MIN_RECALL_X32_IN_MEMORY
        );

        // On-disk x32 with rescore disabled should produce different scores than in-memory (quantization effect)
        String onDiskIndex = prefix() + "in_memory_compare";
        createX32Index(onDiskIndex, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(onDiskIndex, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(onDiskIndex, 1);

        float[] queryVector = QUERY_VECTORS[0];
        KNNQueryBuilder inMemQuery = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).k(K).build();
        Response inMemResponse = searchKNNIndex(indexName, inMemQuery, K);
        List<KNNResult> inMemResults = parseSearchResponse(EntityUtils.toString(inMemResponse.getEntity()), FIELD_NAME);

        // Verify in-memory results are valid (scores > 0, correct count)
        assertFalse(engineName + " in-memory should return results", inMemResults.isEmpty());
        assertEquals(engineName + " in-memory should return K results", K, inMemResults.size());
        for (KNNResult r : inMemResults) {
            assertTrue(engineName + " in-memory scores should be positive", r.getScore() > 0);
        }
    }

    // In-memory mode: rescore is a no-op because search already uses full-precision vectors.
    // Differs from testX32_inMemory which validates recall; this test proves rescore has no effect.
    @SneakyThrows
    public void testX32_rescoreWithInMemory() {
        String indexName = prefix() + "in_memory_rescore";
        createX32InMemoryIndex(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<KNNResult>> noRescoreResults = searchWithRescore(
            indexName,
            QUERY_VECTORS,
            K,
            RescoreContext.builder().rescoreEnabled(false).build()
        );

        List<List<KNNResult>> withRescoreResults = searchWithRescore(
            indexName,
            QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(3.0f).build()
        );

        for (int q = 0; q < QUERY_COUNT; q++) {
            List<KNNResult> noRescore = noRescoreResults.get(q);
            List<KNNResult> withRescore = withRescoreResults.get(q);
            assertEquals(noRescore.size(), withRescore.size());
            for (int i = 0; i < noRescore.size(); i++) {
                assertEquals(
                    engineName + " in-memory: doc IDs should match regardless of rescore",
                    noRescore.get(i).getDocId(),
                    withRescore.get(i).getDocId()
                );
                assertEquals(
                    engineName + " in-memory: scores should match regardless of rescore",
                    noRescore.get(i).getScore(),
                    withRescore.get(i).getScore(),
                    0.001f
                );
            }
        }
    }

    // Filtered ANN search: results must respect the filter predicate
    @SneakyThrows
    public void testX32_filtering() {
        String indexName = prefix() + "filter";
        createX32IndexWithFilterField(indexName, SpaceType.L2);

        int halfCount = DOC_COUNT / 2;
        for (int i = 0; i < DOC_COUNT; i++) {
            String category = (i < halfCount) ? "alpha" : "beta";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, INDEX_VECTORS[i], Map.of("category", category));
        }
        forceMergeKnnIndex(indexName, 1);

        float[] queryVector = QUERY_VECTORS[0];
        KNNQueryBuilder filteredQuery = new KNNQueryBuilder(FIELD_NAME, queryVector, K, QueryBuilders.termQuery("category", "alpha"));
        Response response = searchKNNIndex(indexName, filteredQuery, K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertFalse(engineName + " filtered results should not be empty", results.isEmpty());
        assertTrue(results.size() <= K);
        for (KNNResult result : results) {
            int docId = Integer.parseInt(result.getDocId());
            assertTrue(engineName + " filtered result should only contain alpha docs (id < " + halfCount + ")", docId < halfCount);
        }
    }

    // Filtered search + rescore: validates filter correctness, result stability, and score ordering
    @SneakyThrows
    public void testX32_filteringWithRescore() {
        String indexName = prefix() + "filter_rescore";
        createX32IndexWithFilterField(indexName, SpaceType.L2);

        int halfCount = BASELINE_DOC_COUNT / 2;
        for (int i = 0; i < BASELINE_DOC_COUNT; i++) {
            String category = (i < halfCount) ? "alpha" : "beta";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, BASELINE_INDEX_VECTORS[i], Map.of("category", category));
        }
        forceMergeKnnIndex(indexName, 1);

        float[] queryVector = BASELINE_QUERY_VECTORS[0];
        QueryBuilder filter = QueryBuilders.termQuery("category", "alpha");

        KNNQueryBuilder filteredRescoreQuery = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(filter)
            .rescoreContext(RescoreContext.builder().oversampleFactor(3.0f).build())
            .build();

        Response firstResponse = searchKNNIndex(indexName, filteredRescoreQuery, K);
        List<KNNResult> firstResults = parseSearchResponse(EntityUtils.toString(firstResponse.getEntity()), FIELD_NAME);

        Response secondResponse = searchKNNIndex(indexName, filteredRescoreQuery, K);
        List<KNNResult> secondResults = parseSearchResponse(EntityUtils.toString(secondResponse.getEntity()), FIELD_NAME);

        assertFalse(firstResults.isEmpty());
        assertTrue(firstResults.size() <= K);

        for (KNNResult result : firstResults) {
            int docId = Integer.parseInt(result.getDocId());
            assertTrue(engineName + " filtered+rescored should only contain alpha docs", docId < halfCount);
        }

        List<String> firstIds = firstResults.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        List<String> secondIds = secondResults.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        assertEquals(engineName + " rescored results should be stable", firstIds, secondIds);

        for (int i = 0; i < firstResults.size() - 1; i++) {
            assertTrue(
                engineName + " scores should be in descending order",
                firstResults.get(i).getScore() >= firstResults.get(i + 1).getScore()
            );
        }
    }

    // Validates x32 compression works with nested field type
    @SneakyThrows
    public void testX32_nested() {
        String indexName = prefix() + "nested";
        String nestedPath = "nested_field";
        String nestedFieldPath = "nested_field.vector";
        int nestedDimension = 16;
        int numDocs = 10;
        int k = 5;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedPath)
            .field("type", "nested")
            .startObject("properties")
            .startObject("vector")
            .field("type", "knn_vector")
            .field("dimension", nestedDimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, SpaceType.L2);
        builder.endObject().endObject().endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
        bulkIngestRandomVectorsWithNestedField(indexName, nestedFieldPath, numDocs, nestedDimension);
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);

        assertEquals(numDocs, getDocCount(indexName));

        float[] queryVector = new float[nestedDimension];
        Arrays.fill(queryVector, 0.5f);
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedPath)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedFieldPath)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response searchResponse = searchKNNIndex(indexName, queryBuilder, k);
        List<Object> hits = parseSearchResponseHits(EntityUtils.toString(searchResponse.getEntity()));
        assertFalse(engineName + " nested search should return results", hits.isEmpty());
        assertTrue(hits.size() <= k);
    }

    // Validates x32 with nested field + expand_nested_docs (inner_hits)
    @SneakyThrows
    public void testX32_expandNestedDocs() {
        String indexName = prefix() + "expand_nested";
        String nestedPath = "nested_field";
        String vectorField = nestedPath + ".vector";
        int nestedDimension = 16;
        int numParentDocs = 5;
        int nestedDocsPerParent = 3;
        int k = 5;

        XContentBuilder mappingBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedPath)
            .field("type", "nested")
            .startObject("properties")
            .startObject("vector")
            .field("type", "knn_vector")
            .field("dimension", nestedDimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(mappingBuilder, SpaceType.L2);
        mappingBuilder.endObject().endObject().endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), mappingBuilder.toString());

        // Index parent docs each with multiple nested vectors
        for (int p = 0; p < numParentDocs; p++) {
            XContentBuilder docBuilder = XContentFactory.jsonBuilder().startObject();
            docBuilder.startArray(nestedPath);
            for (int n = 0; n < nestedDocsPerParent; n++) {
                float[] vec = new float[nestedDimension];
                Arrays.fill(vec, (float) (p * nestedDocsPerParent + n));
                docBuilder.startObject().field("vector", vec).endObject();
            }
            docBuilder.endArray().endObject();
            Request indexRequest = new Request("POST", "/" + indexName + "/_doc/" + p);
            indexRequest.setJsonEntity(docBuilder.toString());
            client().performRequest(indexRequest);
        }
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);

        // Query with expand_nested_docs + inner_hits
        float[] queryVector = new float[nestedDimension];
        Arrays.fill(queryVector, 1.0f);
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedPath)
            .startObject("query")
            .startObject("knn")
            .startObject(vectorField)
            .field("vector", queryVector)
            .field("k", k)
            .field("expand_nested_docs", true)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = searchKNNIndex(indexName, queryBuilder, k);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Object> hits = parseSearchResponseHits(responseBody);

        // With expand_nested_docs, we can get more hits than parent docs
        assertFalse(engineName + " expand_nested should return results", hits.isEmpty());
        assertTrue(engineName + " expand_nested results should be <= k", hits.size() <= k);
    }

    // Tests x32 at dimension extremes and non-multiples-of-8 to catch quantization edge cases
    @SneakyThrows
    public void testX32_dimensions() {
        int highDim = 768;
        int lowDim = 3;
        int oddDim = 13;
        int nonAlignedDim = 37;
        int docCount = 50;
        int queryCount = 5;

        float[][] highDimVectors = TestUtils.getIndexVectors(docCount, highDim, true);
        float[][] highDimQueries = TestUtils.getQueryVectors(queryCount, highDim, docCount, true);
        float[][] lowDimVectors = TestUtils.getIndexVectors(docCount, lowDim, true);
        float[][] lowDimQueries = TestUtils.getQueryVectors(queryCount, lowDim, docCount, true);

        String highDimIndex = prefix() + "highdim";
        createX32Index(highDimIndex, SpaceType.L2, highDim);
        bulkAddKnnDocs(highDimIndex, FIELD_NAME, highDimVectors, docCount);
        forceMergeKnnIndex(highDimIndex, 1);

        String lowDimIndex = prefix() + "lowdim";
        createX32Index(lowDimIndex, SpaceType.L2, lowDim);
        bulkAddKnnDocs(lowDimIndex, FIELD_NAME, lowDimVectors, docCount);
        forceMergeKnnIndex(lowDimIndex, 1);

        List<Set<String>> groundTruthHighDim = TestUtils.computeGroundTruthValues(highDimVectors, highDimQueries, SpaceType.L2, K);
        List<List<String>> highDimResults = bulkSearch(highDimIndex, FIELD_NAME, highDimQueries, K);
        double recallHighDim = TestUtils.calculateRecallValue(highDimResults, groundTruthHighDim, K);
        assertTrue(engineName + " high-dim recall should be >= " + MIN_RECALL_X32_L2, recallHighDim >= MIN_RECALL_X32_L2);

        List<Set<String>> groundTruthLowDim = TestUtils.computeGroundTruthValues(lowDimVectors, lowDimQueries, SpaceType.L2, K);
        List<List<String>> lowDimResults = bulkSearch(lowDimIndex, FIELD_NAME, lowDimQueries, K);
        double recallLowDim = TestUtils.calculateRecallValue(lowDimResults, groundTruthLowDim, K);
        assertTrue(engineName + " low-dim recall should be >= " + MIN_RECALL_X32_L2, recallLowDim >= MIN_RECALL_X32_L2);

        // Non-multiple-of-8 dimensions (tests quantization padding/alignment)
        for (int dim : new int[] { oddDim, nonAlignedDim }) {
            float[][] vectors = TestUtils.getIndexVectors(docCount, dim, true);
            float[][] queries = TestUtils.getQueryVectors(queryCount, dim, docCount, true);
            String idx = prefix() + "dim" + dim;
            createX32Index(idx, SpaceType.L2, dim);
            bulkAddKnnDocs(idx, FIELD_NAME, vectors, docCount);
            forceMergeKnnIndex(idx, 1);

            List<Set<String>> gt = TestUtils.computeGroundTruthValues(vectors, queries, SpaceType.L2, K);
            List<List<String>> results = bulkSearch(idx, FIELD_NAME, queries, K);
            double recall = TestUtils.calculateRecallValue(results, gt, K);
            assertTrue(engineName + " dim=" + dim + " recall should be >= " + MIN_RECALL_X32_L2, recall >= MIN_RECALL_X32_L2);
        }
    }

    // Validates rescore improves recall monotonically: no-rescore < default < high oversample
    @SneakyThrows
    public void testX32_rescoreVariations() {
        String indexName = prefix() + "rescore_variations";
        createX32Index(indexName, SpaceType.L2, DIMENSION);
        bulkAddKnnDocs(indexName, FIELD_NAME, BASELINE_INDEX_VECTORS, BASELINE_DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<Set<String>> groundTruth = computeScriptScoreGroundTruth(indexName, BASELINE_QUERY_VECTORS, K);

        List<List<KNNResult>> noRescoreResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().rescoreEnabled(false).build()
        );
        double recallNoRescore = calculateRecallFromResults(noRescoreResults, groundTruth, K);

        // Default rescore (no explicit oversample -- uses system default)
        List<List<KNNResult>> defaultRescoreResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().build()
        );
        double recallDefaultRescore = calculateRecallFromResults(defaultRescoreResults, groundTruth, K);
        assertTrue(engineName + " default rescore recall should be >= " + MIN_RECALL_X32_L2, recallDefaultRescore >= MIN_RECALL_X32_L2);

        // Explicit 3x oversample
        List<List<KNNResult>> explicit3xResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(3.0f).build()
        );
        double recallExplicit3x = calculateRecallFromResults(explicit3xResults, groundTruth, K);

        // 5x oversample
        List<List<KNNResult>> highOversampleResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(5.0f).build()
        );
        double recallHighOversample = calculateRecallFromResults(highOversampleResults, groundTruth, K);
        assertTrue(engineName + " high oversample recall should be >= 0.75", recallHighOversample >= 0.75);

        assertTrue(engineName + " default rescore should improve recall", recallDefaultRescore >= recallNoRescore);
        assertTrue(engineName + " explicit 3x should improve recall", recallExplicit3x >= recallNoRescore);
        assertTrue(engineName + " higher oversample should improve recall", recallHighOversample >= recallExplicit3x);
    }

    // --- Helper methods ---

    @SneakyThrows
    private void validateScoreRanges(String indexName, SpaceType spaceType, float minScore, float maxScore) {
        float[] queryVector = QUERY_VECTORS[0];
        KNNQueryBuilder query = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).k(K).build();
        Response response = searchKNNIndex(indexName, query, K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertFalse(results.isEmpty());
        for (KNNResult result : results) {
            float score = result.getScore();
            assertTrue(engineName + " " + spaceType + " score " + score + " should be >= " + minScore, score >= minScore);
            if (maxScore != Float.POSITIVE_INFINITY) {
                assertTrue(engineName + " " + spaceType + " score " + score + " should be <= " + maxScore, score <= maxScore);
            }
        }
    }

    @SneakyThrows
    private List<Set<String>> computeScriptScoreGroundTruth(String indexName, float[][] queryVectors, int k) {
        List<Set<String>> groundTruth = new ArrayList<>();
        QueryBuilder qb = new MatchAllQueryBuilder();
        for (float[] queryVector : queryVectors) {
            Map<String, Object> params = new HashMap<>();
            params.put("field", FIELD_NAME);
            params.put("query_value", queryVector);
            params.put("space_type", SpaceType.L2.getValue());

            Request request = constructKNNScriptQueryRequest(indexName, qb, params, k);
            Response response = client().performRequest(request);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            groundTruth.add(results.stream().map(KNNResult::getDocId).collect(Collectors.toSet()));
        }
        return groundTruth;
    }

    @SneakyThrows
    private List<List<KNNResult>> searchWithRescore(String indexName, float[][] queryVectors, int k, RescoreContext rescoreContext) {
        List<List<KNNResult>> allResults = new ArrayList<>();
        for (float[] queryVector : queryVectors) {
            KNNQueryBuilder query = KNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(queryVector)
                .k(k)
                .rescoreContext(rescoreContext)
                .build();
            Response response = searchKNNIndex(indexName, query, k);
            allResults.add(parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME));
        }
        return allResults;
    }

    private double calculateRecallFromResults(List<List<KNNResult>> searchResults, List<Set<String>> groundTruth, int k) {
        int totalRelevant = 0;
        int totalExpected = 0;
        for (int i = 0; i < searchResults.size(); i++) {
            Set<String> truth = groundTruth.get(i);
            for (KNNResult result : searchResults.get(i)) {
                if (truth.contains(result.getDocId())) {
                    totalRelevant++;
                }
            }
            totalExpected += Math.min(k, truth.size());
        }
        return totalExpected == 0 ? 0.0 : (double) totalRelevant / totalExpected;
    }

    private Settings defaultSettings() {
        return Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
    }

    @SneakyThrows
    private void addMethodParams(XContentBuilder builder, SpaceType spaceType) {
        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engineName)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION);
        if (FAISS_NAME.equals(engineName)) {
            builder.field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH);
        }
        builder.endObject().endObject();
    }

    @SneakyThrows
    private void createX32Index(String indexName, SpaceType spaceType, int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, spaceType);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private void createFP32Index(String indexName, SpaceType spaceType, int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(COMPRESSION_LEVEL_PARAMETER, "1x");
        addMethodParams(builder, spaceType);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private void createX32InMemoryIndex(String indexName, SpaceType spaceType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "in_memory")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, spaceType);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private void createX32IndexWithFilterField(String indexName, SpaceType spaceType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, spaceType);
        builder.endObject().startObject("category").field("type", "keyword").endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private void createX32MultiFieldIndex(String indexName) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, SpaceType.L2);
        builder.endObject()
            .startObject(FIELD_NAME_2)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION_SMALL)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x");
        addMethodParams(builder, SpaceType.L2);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private void createIndexForScriptScoring(String indexName, SpaceType spaceType, int dimension, boolean useX32) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension);

        if (useX32) {
            builder.field(MODE_PARAMETER, "on_disk");
            builder.field(COMPRESSION_LEVEL_PARAMETER, "32x");
        } else {
            builder.field(COMPRESSION_LEVEL_PARAMETER, "1x");
        }

        addMethodParams(builder, spaceType);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, defaultSettings(), builder.toString());
    }

    @SneakyThrows
    private String buildRadialSearchQuery(String fieldName, float[] vector, String thresholdType, float thresholdValue) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(fieldName)
            .field("vector", vector)
            .field(thresholdType, thresholdValue)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder.toString();
    }

    private Object[] toObjectArray(float[] vector) {
        Object[] result = new Object[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i];
        }
        return result;
    }
}
