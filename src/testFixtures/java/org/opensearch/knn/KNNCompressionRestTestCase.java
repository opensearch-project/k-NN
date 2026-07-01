/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.index.SpaceType;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
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
 * Base class for integration tests that need to run across different compression levels.
 * Provides parameterization infrastructure, helper methods for index creation with compression,
 * and assertion helpers that adapt to compression-specific thresholds.
 */
public abstract class KNNCompressionRestTestCase extends KNNRestTestCase {

    private static final int DEFAULT_HNSW_M = 16;
    private static final int DEFAULT_HNSW_EF_CONSTRUCTION = 100;
    private static final int DEFAULT_HNSW_EF_SEARCH = 100;

    protected final CompressionTestConfig compressionConfig;

    public KNNCompressionRestTestCase(CompressionTestConfig compressionConfig) {
        this.compressionConfig = compressionConfig;
    }

    /**
     * Default parameterization that runs tests for both X1 and X32 compression.
     * Subclasses can override this method to filter which compression configs they support.
     */
    @ParametersFactory(argumentFormatting = "compression:%1$s")
    public static Collection<Object[]> compressionParameters() {
        return Arrays.asList(new Object[] { CompressionTestConfig.X1 }, new Object[] { CompressionTestConfig.X32 });
    }

    /**
     * Returns supported compression configurations for this test class.
     * Override this method to restrict which compression levels a test supports.
     * Default: all configurations (NOT_CONFIGURED, X32)
     */
    protected List<CompressionTestConfig> supportedConfigs() {
        return Arrays.asList(CompressionTestConfig.values());
    }

    /**
     * Creates index name prefix that includes compression level for uniqueness
     */
    protected String prefix() {
        String sanitizedTestName = getTestName().toLowerCase().replaceAll("[^a-z0-9]", "_");
        return sanitizedTestName + "_" + compressionConfig.name().toLowerCase() + "_";
    }

    /**
     * Creates an HNSW index with the compression settings from the current test configuration.
     * Uses default HNSW parameters optimized for test performance.
     */
    @SneakyThrows
    protected void createHnswIndex(String indexName, String engine, SpaceType spaceType, int dimension) {
        createHnswIndex(indexName, engine, spaceType, dimension, DEFAULT_HNSW_M, DEFAULT_HNSW_EF_CONSTRUCTION);
    }

    /**
     * Creates an HNSW index with custom HNSW parameters and compression from test configuration.
     */
    @SneakyThrows
    protected void createHnswIndex(String indexName, String engine, SpaceType spaceType, int dimension, int m, int efConstruction) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension);

        addCompressionMappingFields(builder);
        addMethodParams(builder, engine, spaceType, m, efConstruction);
        builder.endObject().endObject().endObject();

        createKnnIndex(indexName, getDefaultCompressionIndexSettings(), builder.toString());
    }

    /**
     * Creates an HNSW index with additional filter fields for filtered search tests.
     */
    @SneakyThrows
    protected void createHnswIndexWithFilterField(String indexName, String engine, SpaceType spaceType, int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension);

        addCompressionMappingFields(builder);
        addMethodParams(builder, engine, spaceType, DEFAULT_HNSW_M, DEFAULT_HNSW_EF_CONSTRUCTION);
        builder.endObject().startObject("category").field("type", "keyword").endObject().endObject().endObject();

        createKnnIndex(indexName, getDefaultCompressionIndexSettings(), builder.toString());
    }

    /**
     * Asserts that recall meets the minimum threshold for the current compression configuration.
     * Uses different thresholds for FP32 vs quantized compression.
     */
    protected void assertRecall(double actualRecall, SpaceType spaceType) {
        float expectedMinRecall = getMinRecallThreshold(spaceType);
        String message = String.format(
            "%s recall should be >= %.3f but was %.3f",
            compressionConfig.name(),
            expectedMinRecall,
            actualRecall
        );
        assertTrue(message, actualRecall >= expectedMinRecall);
    }

    /**
     * Asserts that search scores are within valid ranges for the space type and compression level.
     * FP32 can assert tighter bounds, quantized compression allows broader ranges due to approximation.
     */
    protected void assertScoreInRange(List<Float> scores, SpaceType spaceType) {
        assertFalse("Scores should not be empty", scores.isEmpty());

        for (float score : scores) {
            assertTrue("Score should be positive", score > 0.0f);

            switch (spaceType) {
                case L2:

                    assertTrue("L2 score should be <= 1.0", score <= 1.0f);
                    break;
                case COSINESIMIL:

                    assertTrue("Cosine score should be <= 2.0", score <= 2.0f);
                    break;
                case INNER_PRODUCT:

                    break;
            }
        }

        for (int i = 0; i < scores.size() - 1; i++) {
            assertTrue("Scores should be in descending order", scores.get(i) >= scores.get(i + 1));
        }
    }

    /**
     * Returns minimum recall threshold based on compression config and space type.
     * Quantized compression generally has lower recall due to approximation.
     */
    protected float getMinRecallThreshold(SpaceType spaceType) {
        if (compressionConfig == CompressionTestConfig.X1) {
            return 0.95f;
        }

        switch (spaceType) {
            case L2:
            case COSINESIMIL:
                return 0.70f;
            case INNER_PRODUCT:
                return 0.60f;
            default:
                return 0.70f;
        }
    }

    /**
     * Returns whether the current compression configuration supports script scoring.
     * All configurations support it since script scoring uses stored vectors, not quantized index.
     */
    protected boolean isScriptScoringSupported() {
        return true;
    }

    /**
     * Returns whether radial search is supported for current compression configuration.
     * Both FP32 and quantized support radial search.
     */
    protected boolean isRadialSearchSupported() {
        return true;
    }

    /**
     * Default index settings optimized for compression testing.
     */
    protected Settings getDefaultCompressionIndexSettings() {
        return Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
    }

    /**
     * Writes the compression-related mapping fields (compression_level, and mode when compressed)
     * into an open knn_vector field object so existing custom mappings can adopt the current config.
     */
    @SneakyThrows
    protected void addCompressionMappingFields(XContentBuilder builder) {
        builder.field(COMPRESSION_LEVEL_PARAMETER, compressionConfig.getCompressionLevelName());
        if (compressionConfig.isCompressed()) {
            builder.field(MODE_PARAMETER, compressionConfig.getModeName());
        }
    }

    /**
     * Applies the compression-related fields (compression_level and mode when compressed) to a
     * {@link KNNJsonIndexMappingsBuilder} builder. Returns the same builder so it can be chained
     * fluently. Injects nothing for the NOT_CONFIGURED configuration.
     */
    protected KNNJsonIndexMappingsBuilder.KNNJsonIndexMappingsBuilderBuilder addCompressionMappingFields(
        KNNJsonIndexMappingsBuilder.KNNJsonIndexMappingsBuilderBuilder builder
    ) {
        if (compressionConfig.isCompressed()) {
            builder.compressionLevel(compressionConfig.getCompressionLevelName());
            builder.mode(compressionConfig.getModeName());
        }
        return builder;
    }

    @SneakyThrows
    private void addMethodParams(XContentBuilder builder, String engine, SpaceType spaceType, int m, int efConstruction) {
        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, m)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);

        if (FAISS_NAME.equals(engine)) {
            builder.field(METHOD_PARAMETER_EF_SEARCH, DEFAULT_HNSW_EF_SEARCH);
        }

        builder.endObject().endObject();
    }
}
