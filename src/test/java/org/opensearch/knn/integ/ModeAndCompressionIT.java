/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import java.nio.charset.StandardCharsets;

import java.util.Locale;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Assert;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.parser.RescoreParser;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.MAPPING_COMPRESSION_NAMES_ARRAY;

public class ModeAndCompressionIT extends KNNRestTestCase {

    private static final String TRAINING_INDEX_NAME = "training_index";
    private static final String TRAINING_FIELD_NAME = "training_field";
    private static final int TRAINING_VECS = 1100;

    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 20;
    private static final int K = NUM_DOCS;
    private final static float[] TEST_VECTOR = new float[] {
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

    private static final String[] COMPRESSION_LEVELS = new String[] {
        CompressionLevel.x2.getName(),
        CompressionLevel.x4.getName(),
        CompressionLevel.x8.getName(),
        CompressionLevel.x16.getName(),
        CompressionLevel.x32.getName() };

    private static final String RE_SCORING_TEST_INDEX = "rescoring-test-index";
    private static final SpaceType RE_SCORING_SPACE_TYPE = SpaceType.INNER_PRODUCT;
    private static final List<Integer> RE_SCORING_DIMENSIONS = Arrays.asList(768, 1000, 1024);

    @SneakyThrows
    public void testIndexCreation_whenInvalid_thenFail() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, "byte")
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .endObject()
            .endObject()
            .endObject();
        String mapping2 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping2));

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "8x")
            .endObject()
            .endObject()
            .endObject();
        String mapping3 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping3));

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk1222")
            .endObject()
            .endObject()
            .endObject();
        String mapping4 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping4));
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testIndexCreation_whenValid_ThenSucceed() {
        XContentBuilder builder;
        for (String compressionLevel : COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + compressionLevel;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            logger.info("Compression level {}", compressionLevel);
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                compressionLevel,
                Mode.NOT_CONFIGURED.getName()
            );
        }

        for (String compressionLevel : COMPRESSION_LEVELS) {
            for (String mode : Mode.NAMES_ARRAY) {
                String indexName = INDEX_NAME + compressionLevel + "_" + mode;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .field(MODE_PARAMETER, mode)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndex(indexName, mapping);
                logger.info("Compression level {}", compressionLevel);
                validateSearch(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    compressionLevel,
                    mode
                );
            }
        }

        for (String mode : Mode.NAMES_ARRAY) {
            String indexName = INDEX_NAME + mode;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(MODE_PARAMETER, mode)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            logger.info("Compression level {}", CompressionLevel.NOT_CONFIGURED.getName());
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                CompressionLevel.NOT_CONFIGURED.getName(),
                mode
            );
        }
    }

    @SneakyThrows
    public void testLowDimensionCompression_whenValid_ThenSucceed() {
        int[] testDimensions = { 2, 5, 12 };

        XContentBuilder builder;
        for (int dimension : testDimensions) {
            for (String compressionLevel : COMPRESSION_LEVELS) {
                String indexName = INDEX_NAME + "_dim" + dimension + "_" + compressionLevel;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", dimension)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndexWithDimension(indexName, mapping, dimension);
                logger.info("Dimension {} with compression level {}", dimension, compressionLevel);
                validateSearchWithDimension(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    compressionLevel,
                    Mode.NOT_CONFIGURED.getName(),
                    dimension
                );
            }

            for (String compressionLevel : COMPRESSION_LEVELS) {
                for (String mode : Mode.NAMES_ARRAY) {
                    String indexName = INDEX_NAME + "_dim" + dimension + "_" + compressionLevel + "_" + mode;
                    builder = XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject("properties")
                        .startObject(FIELD_NAME)
                        .field("type", "knn_vector")
                        .field("dimension", dimension)
                        .field(MODE_PARAMETER, mode)
                        .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                        .endObject()
                        .endObject()
                        .endObject();
                    String mapping = builder.toString();
                    validateIndexWithDimension(indexName, mapping, dimension);
                    logger.info("Dimension {} with compression level {} and mode {}", dimension, compressionLevel, mode);
                    validateSearchWithDimension(
                        indexName,
                        METHOD_PARAMETER_EF_SEARCH,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                        compressionLevel,
                        mode,
                        dimension
                    );
                }
            }

            for (String mode : Mode.NAMES_ARRAY) {
                String indexName = INDEX_NAME + "_dim" + dimension + "_" + mode;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", dimension)
                    .field(MODE_PARAMETER, mode)
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndexWithDimension(indexName, mapping, dimension);
                logger.info(
                    "Dimension {} with mode {} and compression level {}",
                    dimension,
                    mode,
                    CompressionLevel.NOT_CONFIGURED.getName()
                );
                validateSearchWithDimension(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    CompressionLevel.NOT_CONFIGURED.getName(),
                    mode,
                    dimension
                );
            }
        }
    }

    @SneakyThrows
    public void testQueryRescoreEnabledAndDisabled() {
        XContentBuilder builder;
        String mode = Mode.ON_DISK.getName();
        String compressionLevel = CompressionLevel.x32.getName();
        String indexName = INDEX_NAME + compressionLevel;
        // Explicitly use binary encoder to test BQ rescoring behavior
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, mode)
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
            .startObject(KNN_METHOD)
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "binary")
            .startObject(PARAMETERS)
            .field("bits", 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        validateIndex(indexName, mapping);
        logger.info("Compression level {}", compressionLevel);
        // Do exact search and gather right scores for the documents
        Response exactSearchResponse = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("script_score")
                .startObject("query")
                .field("match_all")
                .startObject()
                .endObject()
                .endObject()
                .startObject("script")
                .field("source", "knn_score")
                .field("lang", "knn")
                .startObject("params")
                .field("field", FIELD_NAME)
                .field("query_value", TEST_VECTOR)
                .field("space_type", SpaceType.L2.getValue())
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(exactSearchResponse);
        String exactSearchResponseBody = EntityUtils.toString(exactSearchResponse.getEntity());
        List<Float> exactSearchKnnResults = parseSearchResponseScore(exactSearchResponseBody, FIELD_NAME);
        assertEquals(NUM_DOCS, exactSearchKnnResults.size());
        // Search without rescore
        Response response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .field(RescoreParser.RESCORE_PARAMETER, false)
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Float> knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        Assert.assertNotEquals(exactSearchKnnResults, knnResults);
        // Search with explicit rescore
        response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .startObject(RescoreParser.RESCORE_PARAMETER)
                .field(RescoreParser.RESCORE_OVERSAMPLE_PARAMETER, 2.0f)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        Assert.assertEquals(exactSearchKnnResults, knnResults);
        // Search with default rescore
        response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        Assert.assertEquals(exactSearchKnnResults, knnResults);
    }

    @SneakyThrows
    public void testDefaultRescoringEnabled_whenForDifferentDimensionThresholds_thenSuccess() {
        for (int dim : RE_SCORING_DIMENSIONS) {
            String index = RE_SCORING_TEST_INDEX + "-dim-" + dim;
            createOnDiskIndex(index, dim, RE_SCORING_SPACE_TYPE);
            float[][] vectors = new float[1][dim];
            for (int i = 0; i < dim; i++) {
                vectors[0][i] = 2;
            }
            bulkAddKnnDocs(index, FIELD_NAME, vectors, vectors.length);
            refreshIndex(index);
            float[] query = new float[dim];
            Arrays.fill(query, 1);
            String kNNQuery = buildKNNQuery(query);
            String responseString = EntityUtils.toString(performSearch(index, kNNQuery).getEntity());
            Assert.assertEquals(1, parseIds(responseString).size());
            double actualScore = RE_SCORING_SPACE_TYPE.getKnnVectorSimilarityFunction().compare(query, vectors[0]);
            double expectedScore = parseScores(responseString).get(0);
            Assert.assertEquals("Assert Failed for Rescoring test with dimension : " + dim, actualScore, expectedScore, 0);
            deleteKNNIndex(index);
        }
    }

    private String buildKNNQuery(float[] queryVector) throws IOException {
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(FIELD_NAME);
        queryBuilder.field("vector", queryVector);
        queryBuilder.field("k", 10);
        queryBuilder.endObject().endObject().endObject().endObject();
        return queryBuilder.toString();
    }

    @SneakyThrows
    public void testDeletedDocsWithSegmentMerge_whenValid_ThenSucceed() {
        XContentBuilder builder;
        CompressionLevel compressionLevel = CompressionLevel.x32;
        String indexName = INDEX_NAME + compressionLevel;
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        validateIndexWithDeletedDocs(indexName, mapping);
        validateGreenIndex(indexName);
    }

    @SneakyThrows
    public void testCompressionIndexWithNonVectorFieldsSegment_whenValid_ThenSucceed() {
        CompressionLevel compressionLevel = CompressionLevel.x32;
        String indexName = INDEX_NAME + compressionLevel;
        try (
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                .endObject()
                .endObject()
                .endObject()
        ) {
            String mapping = builder.toString();
            Settings indexSettings = buildKNNIndexSettings(0);
            createKnnIndex(indexName, indexSettings, mapping);
            // since we are going to delete a document, so its better to have 1 more extra doc so that we can re-use some tests
            addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS + 1);
            addNonKNNDoc(indexName, String.valueOf(NUM_DOCS + 2), FIELD_NAME_NON_KNN, "Hello world");
            // Delete the last doc (furthest from query vector) to avoid SQ 1-bit's better recall
            // pulling a deleted doc into top-k results
            deleteKnnDoc(indexName, String.valueOf(NUM_DOCS));
            flushIndex(indexName);
            validateGreenIndex(indexName);
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                compressionLevel.getName(),
                Mode.ON_DISK.getName()
            );
        }
    }

    /**
     * Regression test for merging a persisted empty FAISS 1-bit scalar-quantized segment with a
     * segment containing vectors.
     */
    @SneakyThrows
    public void testForceMergeWithVectorlessSegment_whenFaissSQ_thenSucceed() {
        final int vectorDocCount = 10;
        final int k = 5;
        final String indexName = INDEX_NAME + "_sq_empty_merge";
        final float[][] vectors = new float[vectorDocCount][DIMENSION];
        for (int docId = 0; docId < vectorDocCount; docId++) {
            Arrays.fill(vectors[docId], (float) docId);
        }

        try (
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject(KNN_METHOD)
                .field(NAME, "hnsw")
                .field(KNN_ENGINE, FAISS_NAME)
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "sq")
                .startObject(PARAMETERS)
                .field("bits", 1)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .startObject(FIELD_NAME_NON_KNN)
                .field("type", "text")
                .endObject()
                .endObject()
                .endObject()
        ) {
            final Settings indexSettings = Settings.builder()
                .put(buildKNNIndexSettings(-1))
                .put("index.soft_deletes.retention_lease.period", "0ms")
                .build();
            createKnnIndex(indexName, indexSettings, builder.toString());

            // Keep a live non-vector document in the original vector-bearing segment. This prevents
            // Lucene from dropping the segment when every vector document is replaced below.
            final Request initialBulk = new Request("POST", "/_bulk");
            initialBulk.addParameter("refresh", "true");
            final StringBuilder initialBody = new StringBuilder();
            for (int docId = 0; docId < vectorDocCount; docId++) {
                initialBody.append("{\"index\":{\"_index\":\"")
                    .append(indexName)
                    .append("\",\"_id\":\"")
                    .append(docId)
                    .append("\"}}\n{\"")
                    .append(FIELD_NAME)
                    .append("\":")
                    .append(Arrays.toString(vectors[docId]))
                    .append("}\n");
            }
            initialBody.append("{\"index\":{\"_index\":\"")
                .append(indexName)
                .append("\",\"_id\":\"anchor\"}}\n{\"")
                .append(FIELD_NAME_NON_KNN)
                .append("\":\"anchor\"}\n");
            initialBulk.setJsonEntity(initialBody.toString());
            assertEquals(RestStatus.OK.getStatus(), client().performRequest(initialBulk).getStatusLine().getStatusCode());
            assertEquals(1, getTotalSegmentCount(indexName));

            final Request replacementBulk = new Request("POST", "/_bulk");
            replacementBulk.addParameter("refresh", "true");
            final StringBuilder replacementBody = new StringBuilder();
            for (int docId = 0; docId < vectorDocCount; docId++) {
                replacementBody.append("{\"index\":{\"_index\":\"")
                    .append(indexName)
                    .append("\",\"_id\":\"")
                    .append(docId)
                    .append("\"}}\n{\"")
                    .append(FIELD_NAME_NON_KNN)
                    .append("\":\"replacement\"}\n");
            }
            replacementBulk.setJsonEntity(replacementBody.toString());
            assertEquals(RestStatus.OK.getStatus(), client().performRequest(replacementBulk).getStatusLine().getStatusCode());
            assertEquals(2, getTotalSegmentCount(indexName));

            // Reopen with zero lease retention so the next merge can drop the soft-deleted vector
            // documents and persist SQ metadata with zero vectors.
            flushIndex(indexName);
            closeKNNIndex(indexName);
            openIndex(indexName);
            validateGreenIndex(indexName);
            forceMergeKnnIndex(indexName, 1);
            assertEquals(vectorDocCount + 1, getDocCount(indexName));
            assertEquals(1, getTotalSegmentCount(indexName));

            bulkAddKnnDocs(indexName, FIELD_NAME, vectors, vectorDocCount);

            // This merge reads the persisted empty SQ segment. Without the reader-side empty-values
            // guard it fails with NoSuchFieldException for Lucene's quantizedVectorValues field.
            forceMergeKnnIndex(indexName, 1);
            assertEquals(vectorDocCount + 1, getDocCount(indexName));
            assertEquals(1, getTotalSegmentCount(indexName));
            validateKNNSearch(indexName, FIELD_NAME, DIMENSION, vectorDocCount, k);
        }
    }

    /**
     * Regression test for the missing {@code .osknnqstate} quantization-state failure on the k-NN
     * <b>search</b> read path (see {@code repro_osknnqstate_missing.sh}).
     * <p>
     * A FAISS binary (1-bit) quantized field can end up in a segment's FieldInfos while that segment
     * holds zero live vector documents (all vector docs deleted then physically expunged by a merge).
     * The writer skips {@code train()} for such a segment, so no {@code .osknnqstate} file is written.
     * The reader {@link org.opensearch.knn.index.query.SegmentLevelQuantizationInfo#build} must detect
     * the empty segment (via {@code FloatVectorValues}) and return {@code null} instead of
     * unconditionally opening the absent state file, which previously threw
     * {@code NoSuchFileException}/{@code FileNotFoundException} before the graceful exact-search fallback.
     * <p>
     * Unlike {@link #testForceMergeWithVectorlessSegment_whenFaissSQ_thenSucceed}, the vectorless segment
     * is left un-merged alongside a healthy vector-bearing segment so that a live k-NN query touches it.
     */
    @SneakyThrows
    public void testKnnSearchWithVectorlessSegment_whenFaissBinary_thenSucceed() {
        final int vectorDocCount = 10;
        final int k = 5;
        final String indexName = INDEX_NAME + "_bq_empty_search";
        final float[][] vectors = new float[vectorDocCount][DIMENSION];
        for (int docId = 0; docId < vectorDocCount; docId++) {
            Arrays.fill(vectors[docId], (float) docId);
        }

        try (
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject(KNN_METHOD)
                .field(NAME, "hnsw")
                .field(KNN_ENGINE, FAISS_NAME)
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "binary")
                .startObject(PARAMETERS)
                .field("bits", 1)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .startObject(FIELD_NAME_NON_KNN)
                .field("type", "text")
                .endObject()
                .endObject()
                .endObject()
        ) {
            createKnnIndex(indexName, buildKNNIndexSettings(-1), builder.toString());

            // Segment A: vector docs plus one live non-vector "anchor" doc. The anchor keeps the segment
            // alive after every vector doc below is replaced, so the field survives in FieldInfos.
            final Request initialBulk = new Request("POST", "/_bulk");
            initialBulk.addParameter("refresh", "true");
            final StringBuilder initialBody = new StringBuilder();
            for (int docId = 0; docId < vectorDocCount; docId++) {
                initialBody.append("{\"index\":{\"_index\":\"")
                    .append(indexName)
                    .append("\",\"_id\":\"")
                    .append(docId)
                    .append("\"}}\n{\"")
                    .append(FIELD_NAME)
                    .append("\":")
                    .append(Arrays.toString(vectors[docId]))
                    .append("}\n");
            }
            initialBody.append("{\"index\":{\"_index\":\"")
                .append(indexName)
                .append("\",\"_id\":\"anchor\"}}\n{\"")
                .append(FIELD_NAME_NON_KNN)
                .append("\":\"anchor\"}\n");
            initialBulk.setJsonEntity(initialBody.toString());
            assertEquals(RestStatus.OK.getStatus(), client().performRequest(initialBulk).getStatusLine().getStatusCode());
            assertEquals(1, getTotalSegmentCount(indexName));

            // Replace every vector doc with a non-vector doc (soft-deletes the original vectors).
            final Request replacementBulk = new Request("POST", "/_bulk");
            replacementBulk.addParameter("refresh", "true");
            final StringBuilder replacementBody = new StringBuilder();
            for (int docId = 0; docId < vectorDocCount; docId++) {
                replacementBody.append("{\"index\":{\"_index\":\"")
                    .append(indexName)
                    .append("\",\"_id\":\"")
                    .append(docId)
                    .append("\"}}\n{\"")
                    .append(FIELD_NAME_NON_KNN)
                    .append("\":\"replacement\"}\n");
            }
            replacementBulk.setJsonEntity(replacementBody.toString());
            assertEquals(RestStatus.OK.getStatus(), client().performRequest(replacementBulk).getStatusLine().getStatusCode());
            assertEquals(2, getTotalSegmentCount(indexName));

            // Physically expunge the soft-deleted vector docs so the surviving segment keeps the field
            // in FieldInfos with zero live vectors (hence no .osknnqstate is written). Soft-delete
            // retention leases advance in the background, so poll -- flush + expunge-only merge -- until
            // docs.deleted reaches 0 instead of forcing it with a retention-lease override setting.
            assertBusy(() -> {
                flushIndex(indexName);
                final Request expunge = new Request("POST", "/" + indexName + "/_forcemerge");
                expunge.addParameter("only_expunge_deletes", "true");
                expunge.addParameter("flush", "true");
                assertEquals(RestStatus.OK.getStatus(), client().performRequest(expunge).getStatusLine().getStatusCode());
                refreshIndex(indexName);
                assertEquals("soft-deleted vector docs were not physically expunged", 0, getDeletedDocCount(indexName));
            }, 120, TimeUnit.SECONDS);
            assertEquals(vectorDocCount + 1, getDocCount(indexName));

            // Add a fresh, healthy vector-bearing segment. Do NOT merge -- the vectorless segment must
            // remain a separate leaf so the k-NN query below actually reads it and exercises the
            // SegmentLevelQuantizationInfo.build guard.
            bulkAddKnnDocs(indexName, FIELD_NAME, vectors, vectorDocCount);
            // At least two leaves must remain: the vectorless segment plus the fresh vector segment(s).
            // A count of exactly 1 would mean everything was merged into a single healthy segment and the
            // guard would never be exercised.
            assertTrue(getTotalSegmentCount(indexName) >= 2);

            // Before the fix this throws NoSuchFileException on the missing .osknnqstate of the
            // vectorless segment. After the fix the empty segment degrades gracefully and the query
            // returns the healthy segment's neighbors. We assert the full result set is returned (a
            // shard failure would drop hits) rather than a strict neighbor order, since 1-bit binary
            // quantization does not preserve exact ranking.
            final float[] queryVector = new float[DIMENSION];
            Arrays.fill(queryVector, (float) vectorDocCount);
            final Response searchResponse = searchKNNIndex(
                indexName,
                KNNQueryBuilder.builder().k(k).fieldName(FIELD_NAME).vector(queryVector).build(),
                k
            );
            final List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), FIELD_NAME);
            assertEquals(k, results.size());
        }
    }

    /**
     * Number of soft-deleted (but not yet physically purged) docs in the primaries of an index,
     * read from the {@code _stats/docs} API.
     */
    private int getDeletedDocCount(final String indexName) throws Exception {
        final Request request = new Request("GET", "/" + indexName + "/_stats/docs");
        final Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        final Map<String, Object> responseMap = createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            EntityUtils.toString(response.getEntity())
        ).map();
        final Map<String, Object> all = (Map<String, Object>) responseMap.get("_all");
        final Map<String, Object> primaries = (Map<String, Object>) all.get("primaries");
        final Map<String, Object> docs = (Map<String, Object>) primaries.get("docs");
        return (Integer) docs.get("deleted");
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates separate segments: one with vector docs, one with only non-vector doc.
     * Validates k-NN search functionality works without errors for ON_DISK mode with compression.
     */
    @SneakyThrows
    public void testMixedSegmentsWithNonVectorDoc_whenValid_ThenSucceed() {
        for (String compressionLevelName : COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + "_mixed_segments_" + compressionLevelName;
            try (
                XContentBuilder builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevelName)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .startObject(FIELD_NAME_NON_KNN)
                    .field("type", "text")
                    .endObject()
                    .endObject()
                    .endObject()
            ) {
                String mapping = builder.toString();
                Settings indexSettings = buildKNNIndexSettings(0);
                createKnnIndex(indexName, indexSettings, mapping);

                // Add 21 docs with vector fields
                addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS + 1);
                // Flush to create a segment with vector docs
                flush(indexName, true);

                // Add 1 doc with only non-vector field (should get its own segment)
                addNonKNNDoc(indexName, String.valueOf(NUM_DOCS + 2), FIELD_NAME_NON_KNN, "Non-vector document");
                // Flush to ensure proper segmentation
                flush(indexName, true);

                validateGreenIndex(indexName);
                int segmentCount = getTotalSegmentCount(indexName);
                assertTrue(segmentCount >= 2);

                validateSearch(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    compressionLevelName,
                    Mode.ON_DISK.getName()
                );
            }
        }
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates a doc with vector field, then updates it to remove the vector field.
     * Validates k-NN search functionality works without errors for ON_DISK mode with compression.
     */
    @SneakyThrows
    public void testVectorFieldRemovalByUpdate_whenValid_thenSucceed() {
        for (String compressionLevelName : COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + "_vector_removal_" + compressionLevelName;

            try (
                XContentBuilder builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevelName)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .startObject("description")
                    .field("type", "text")
                    .endObject()
                    .endObject()
                    .endObject()
            ) {
                String mapping = builder.toString();
                Settings indexSettings = buildKNNIndexSettings(0);
                createKnnIndex(indexName, indexSettings, mapping);

                // Add 21 docs with vector fields
                addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS + 1);
                // Flush to create a segment with vector docs
                flush(indexName, true);

                // Add doc with both vector and text field
                String docId = String.valueOf(NUM_DOCS + 1);
                String docWithBoth = XContentFactory.jsonBuilder()
                    .startObject()
                    .field(FIELD_NAME, TEST_VECTOR)
                    .field("description", "Test document")
                    .endObject()
                    .toString();
                addKnnDoc(indexName, docId, docWithBoth);

                // Update doc to remove vector field, keeping only text field
                addNonKNNDoc(indexName, docId, "description", "Updated test document");
                // Flush to create a new segment containing doc with no vector fields
                flush(indexName, true);

                validateGreenIndex(indexName);

                validateSearch(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    compressionLevelName,
                    Mode.ON_DISK.getName()
                );
            }
        }
    }

    @SneakyThrows
    public void testTraining_whenInvalid_thenFail() {
        setupTrainingIndex();
        String modelId = "test";

        XContentBuilder builder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
            .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, "float")
            .field(MODEL_DESCRIPTION, "")
            .field(MODE_PARAMETER, Mode.ON_DISK)
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "pq")
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, 2)
            .field(ENCODER_PARAMETER_PQ_M, 8)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        expectThrows(ResponseException.class, () -> trainModel(modelId, builder1));

        XContentBuilder builder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
            .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, "binary")
            .field(MODEL_DESCRIPTION, "")
            .field(MODE_PARAMETER, Mode.ON_DISK)
            .endObject();
        expectThrows(ResponseException.class, () -> trainModel(modelId, builder2));
    }

    @SneakyThrows
    public void testTraining_whenValid_thenSucceed() {
        setupTrainingIndex();
        XContentBuilder builder;
        for (String compressionLevel : MAPPING_COMPRESSION_NAMES_ARRAY) {
            if (compressionLevel.equals("4x")) {
                continue;
            }
            String indexName = INDEX_NAME + compressionLevel;
            String modelId = indexName;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                .field(KNNConstants.DIMENSION, DIMENSION)
                .field(MODEL_DESCRIPTION, "")
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                .endObject();
            validateTraining(modelId, builder);
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("model_id", modelId)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(
                indexName,
                METHOD_PARAMETER_NPROBES,
                METHOD_PARAMETER_NLIST_DEFAULT,
                compressionLevel,
                Mode.NOT_CONFIGURED.getName()
            );
            deleteKNNIndex(indexName);
        }
        for (String mode : Mode.NAMES_ARRAY) {
            if (mode == null) {
                continue;
            }
            mode = mode.toLowerCase(Locale.ROOT);
            String indexName = INDEX_NAME + mode;
            String modelId = indexName;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                .field(KNNConstants.DIMENSION, DIMENSION)
                .field(MODEL_DESCRIPTION, "")
                .field(MODE_PARAMETER, mode)
                .endObject();
            validateTraining(modelId, builder);
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("model_id", modelId)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(
                indexName,
                METHOD_PARAMETER_NPROBES,
                METHOD_PARAMETER_NLIST_DEFAULT,
                CompressionLevel.NOT_CONFIGURED.getName(),
                mode
            );
            deleteKNNIndex(indexName);
        }
    }

    @SneakyThrows
    private void validateIndex(String indexName, String mapping) {
        createKnnIndex(indexName, mapping);
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName, 1);
    }

    @SneakyThrows
    private void validateIndexWithDimension(String indexName, String mapping, int dimension) {
        createKnnIndex(indexName, mapping);
        addKNNDocs(indexName, FIELD_NAME, dimension, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName, 1);
    }

    @SneakyThrows
    private void validateSearchWithDimension(
        String indexName,
        String methodParameterName,
        int methodParameterValue,
        String compressionLevelString,
        String mode,
        int dimension
    ) {
        float[] testVector = new float[dimension];
        Arrays.fill(testVector, 1.0f);

        // Basic search
        Response response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", testVector)
                .field("k", K)
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Float> knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());

        // Do exact search and gather right scores for the documents
        Response exactSearchResponse = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("script_score")
                .startObject("query")
                .field("match_all")
                .startObject()
                .endObject()
                .endObject()
                .startObject("script")
                .field("source", "knn_score")
                .field("lang", "knn")
                .startObject("params")
                .field("field", FIELD_NAME)
                .field("query_value", testVector)
                .field("space_type", SpaceType.L2.getValue())
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(exactSearchResponse);
        String exactSearchResponseBody = EntityUtils.toString(exactSearchResponse.getEntity());
        List<Float> exactSearchKnnResults = parseSearchResponseScore(exactSearchResponseBody, FIELD_NAME);
        assertEquals(NUM_DOCS, exactSearchKnnResults.size());
        if (Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }

        // Search with rescore
        response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", testVector)
                .field("k", K)
                .startObject(RescoreParser.RESCORE_PARAMETER)
                .field(RescoreParser.RESCORE_OVERSAMPLE_PARAMETER, 2.0f)
                .endObject()
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        if (Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }
    }

    @SneakyThrows
    private void validateIndexWithDeletedDocs(String indexName, String mapping) {
        createKnnIndex(indexName, mapping);
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        refreshIndex(indexName);
        // this will simulate the deletion of the docs
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);
        refreshIndex(indexName);
    }

    @SneakyThrows
    private void validateGreenIndex(String indexName) {
        Request request = new Request("GET", "/_cat/indices/" + indexName + "?format=csv");
        Response response = client().performRequest(request);
        assertOK(response);
        assertEquals(
            "The status of index " + indexName + " is not green",
            "green",
            new String(response.getEntity().getContent().readAllBytes(), StandardCharsets.UTF_8).split("\n")[0].split(" ")[0]
        );

    }

    @SneakyThrows
    private void setupTrainingIndex() {
        createBasicKnnIndex(TRAINING_INDEX_NAME, TRAINING_FIELD_NAME, DIMENSION);
        bulkIngestRandomVectors(TRAINING_INDEX_NAME, TRAINING_FIELD_NAME, TRAINING_VECS, DIMENSION);
    }

    @SneakyThrows
    private void validateTraining(String modelId, XContentBuilder builder) {
        Response trainResponse = trainModel(modelId, builder);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
        assertTrainingSucceeds(modelId, 360, 1000);
    }

    @SneakyThrows
    private void validateSearch(
        String indexName,
        String methodParameterName,
        int methodParameterValue,
        String compressionLevelString,
        String mode
    ) {
        // Basic search
        Response response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Float> knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());

        // Do exact search and gather right scores for the documents
        Response exactSearchResponse = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("script_score")
                .startObject("query")
                .field("match_all")
                .startObject()
                .endObject()
                .endObject()
                .startObject("script")
                .field("source", "knn_score")
                .field("lang", "knn")
                .startObject("params")
                .field("field", FIELD_NAME)
                .field("query_value", TEST_VECTOR)
                .field("space_type", SpaceType.L2.getValue())
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(exactSearchResponse);
        String exactSearchResponseBody = EntityUtils.toString(exactSearchResponse.getEntity());
        List<Float> exactSearchKnnResults = parseSearchResponseScore(exactSearchResponseBody, FIELD_NAME);
        assertEquals(NUM_DOCS, exactSearchKnnResults.size());
        if (Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }

        // Search with rescore
        response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .startObject(RescoreParser.RESCORE_PARAMETER)
                .field(RescoreParser.RESCORE_OVERSAMPLE_PARAMETER, 2.0f)
                .endObject()
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        if (Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }
    }
}
