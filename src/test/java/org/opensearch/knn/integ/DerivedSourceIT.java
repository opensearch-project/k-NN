/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.Builder;
import lombok.Data;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.CheckedConsumer;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality. Ignoring tests for now as feature is experimental.
 */
public class DerivedSourceIT extends KNNRestTestCase {

    private final static String NESTED_NAME = "test_nested";
    private final static String FIELD_NAME = "test_vector";
    private final int TEST_DIMENSION = 128;
    private final int DOCS = 50;

    private static final Settings DERIVED_ENABLED_SETTINGS = Settings.builder()
        .put("number_of_shards", 1)
        .put("number_of_replicas", 0)
        .put("index.knn", true)
        .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, true)
        .build();
    private static final Settings DERIVED_DISABLED_SETTINGS = Settings.builder()
        .put("number_of_shards", 1)
        .put("number_of_replicas", 0)
        .put("index.knn", true)
        .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, false)
        .build();

    /**
     * Testing flat, single field base case with index configuration. The test will automatically skip adding fields for
     *  random documents to ensure it works robustly. To ensure correctness, we repeat same operations against an
     *  index without derived source enabled (baseline).
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             }
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFlatBaseCase() {
        String mapping = createVectorNonNestedMappings(TEST_DIMENSION, null);
        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 32, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 32, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Testing flat, single field base case with index configuration. The test will automatically skip adding fields for
     *  random documents to ensure it works robustly. To ensure correctness, we repeat same operations against an
     *  index without derived source enabled (baseline).
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128,
     *                 "data_type": "byte"
     *             }
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128,
     *                 "data_type": "byte"
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFlatByteBaseCase() {
        String mapping = createVectorNonNestedMappings(TEST_DIMENSION, "byte");
        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 8, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 8, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 1))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Testing flat, single field base case with index configuration. The test will automatically skip adding fields for
     *  random documents to ensure it works robustly. To ensure correctness, we repeat same operations against an
     *  index without derived source enabled (baseline).
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128,
     *                 "data_type": "binary"
     *             }
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 128,
     *                 "data_type": "binary"
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFlatBinaryBaseCase() {
        String mapping = createVectorNonNestedMappings(TEST_DIMENSION, "binary");
        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 1, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkips(context.indexName, FIELD_NAME, context.docCount, context.dimension, 1, 0.1f);
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomByteVector(c.dimension, 8))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Testing multiple flat fields.
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector1": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             },
     *             "test_vector1": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             },
     *             "text": {
     *                 "type": "text",
     *             },
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_vector1": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             },
     *             "test_vector1": {
     *                 "type": "knn_vector",
     *                 "dimension": 128
     *             },
     *             "text": {
     *                 "type": "text",
     *             },
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testMultiFlatFields() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME + "1")
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()
            .startObject(FIELD_NAME + "2")
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()
            .startObject("text")
            .field(TYPE, "text")
            .endObject()
            .endObject()
            .endObject();
        String multiFieldMapping = builder.toString();

        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndMultFields(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        context.vectorFieldNames.get(1),
                        "text",
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndMultFields(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        context.vectorFieldNames.get(1),
                        "text",
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME + "1", FIELD_NAME + "2"))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(multiFieldMapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Testing single nested doc per parent doc.
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_nested" : {
     *               "type": "nested",
     *               "properties": {
     *                 "test_vector": {
     *                     "type": "knn_vector",
     *                     "dimension": 128
     *                 },
     *                 "text": {
     *                     "type": "text",
     *                 },
     *               }
     *             }
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_nested" : {
     *               "type": "nested",
     *               "properties": {
     *                 "test_vector": {
     *                     "type": "knn_vector",
     *                     "dimension": 128
     *                 },
     *                 "text": {
     *                     "type": "text",
     *                 },
     *               }
     *             }
     *         }
     *     }
     * }
     */
    public void testNestedSingleDocBasic() {
        String nestedMapping = createVectorNestedMappings(TEST_DIMENSION, null);
        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndNested(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        NESTED_NAME + "." + "text",
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndNested(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        NESTED_NAME + "." + "text",
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Testing single nested doc per parent doc.
     * Test mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_nested" : {
     *               "type": "nested",
     *               "properties": {
     *                 "test_vector": {
     *                     "type": "knn_vector",
     *                     "dimension": 128
     *                 },
     *                 "text": {
     *                     "type": "text",
     *                 },
     *               }
     *             }
     *         }
     *     }
     * }
     * Baseline mapping:
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_nested" : {
     *               "type": "nested",
     *               "properties": {
     *                 "test_vector": {
     *                     "type": "knn_vector",
     *                     "dimension": 128
     *                 },
     *                 "text": {
     *                     "type": "text",
     *                 },
     *               }
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testNestedMultiDocBasic() {
        String nestedMapping = createVectorNestedMappings(TEST_DIMENSION, null);
        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndNestedMultiDoc(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        NESTED_NAME + "." + "text",
                        context.docCount,
                        context.dimension,
                        0.1f,
                        5
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsWithSkipsAndNestedMultiDoc(
                        context.indexName,
                        context.vectorFieldNames.get(0),
                        NESTED_NAME + "." + "text",
                        context.docCount,
                        context.dimension,
                        0.1f,
                        5
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(NESTED_NAME + "." + FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(nestedMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Test object (non-nested field)
     * Test
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true
     *     },
     *     "mappings":{
     *       {
     *           "properties": {
     *               "vector_field_1" : {
     *                 "type" : "knn_vector",
     *                 "dimension" : 2
     *               },
     *               "path_1": {
     *                   "properties" : {
     *                       "vector_field_2" : {
     *                         "type" : "knn_vector",
     *                         "dimension" : 2
     *                       },
     *                       "path_2": {
     *                           "properties" : {
     *                               "vector_field_3" : {
     *                                 "type" : "knn_vector",
     *                                 "dimension" : 2
     *                               },
     *                           }
     *                       }
     *                   }
     *               }
     *           }
     *       }
     *     }
     * }
     * Baseline
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": false
     *     },
     *     "mappings":{
     *       {
     *           "properties": {
     *               "vector_field_1" : {
     *                 "type" : "knn_vector",
     *                 "dimension" : 2
     *               },
     *               "path_1": {
     *                   "properties" : {
     *                       "vector_field_2" : {
     *                         "type" : "knn_vector",
     *                         "dimension" : 2
     *                       },
     *                       "path_2": {
     *                           "properties" : {
     *                               "vector_field_3" : {
     *                                 "type" : "knn_vector",
     *                                 "dimension" : 2
     *                               },
     *                           }
     *                       }
     *                   }
     *               }
     *           }
     *       }
     *     }
     * }
     */
    @SneakyThrows
    public void testObjectFieldTypes() {
        String PATH_1_NAME = "path_1";
        String PATH_2_NAME = "path_2";

        String objectFieldTypeMapping = XContentFactory.jsonBuilder()
            .startObject() // 1-open
            .startObject(PROPERTIES_FIELD) // 2-open
            .startObject(FIELD_NAME + "1")
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()

            .startObject(PATH_1_NAME)
            .startObject(PROPERTIES_FIELD)

            .startObject(FIELD_NAME + "2")
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()
            .startObject(PATH_2_NAME)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME + "3")
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        List<IndexConfigContext> indexConfigContexts = List.of(
            IndexConfigContext.builder()
                .indexName(("original-enable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsMultiFieldsWithSkips(
                        context.indexName,
                        context.vectorFieldNames,
                        List.of("text", PATH_1_NAME + "." + "text", PATH_1_NAME + "." + PATH_2_NAME + "." + "text"),
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("original-disable-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {
                    bulkIngestRandomVectorsMultiFieldsWithSkips(
                        context.indexName,
                        context.vectorFieldNames,
                        List.of("text", PATH_1_NAME + "." + "text", PATH_1_NAME + "." + PATH_2_NAME + "." + "text"),
                        context.docCount,
                        context.dimension,
                        0.1f
                    );
                    refreshAllIndices();
                })
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build(),
            IndexConfigContext.builder()
                .indexName(("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT))
                .vectorFieldNames(
                    List.of(
                        FIELD_NAME + "1",
                        PATH_1_NAME + "." + FIELD_NAME + "2",
                        PATH_1_NAME + "." + PATH_2_NAME + "." + FIELD_NAME + "3"
                    )
                )
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_DISABLED_SETTINGS)
                .mapping(objectFieldTypeMapping)
                .isNested(true)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .updateVectorSupplier((c) -> randomFloatVector(c.dimension))
                .build()

        );
        testDerivedSourceE2E(indexConfigContexts);
    }

    /**
     * Single method for running end to end tests for different index configurations for derived source. In general,
     * flow of operations are
     *
     * @param indexConfigContexts {@link IndexConfigContext}
     */
    @SneakyThrows
    private void testDerivedSourceE2E(List<IndexConfigContext> indexConfigContexts) {
        // Make sure there are 6
        assertEquals(6, indexConfigContexts.size());

        // Prepare the indices by creating them and ingesting data into them
        prepareOriginalIndices(indexConfigContexts);

        // Merging
        testMerging(indexConfigContexts);

        // Update. Skipping update tests for nested docs for now. Will add in the future.
        if (indexConfigContexts.get(0).isNested() == false) {
            testUpdate(indexConfigContexts);
        }

        // Delete
        testDelete(indexConfigContexts);

        // Search
        testSearch(indexConfigContexts);

        // Reindex
        testReindex(indexConfigContexts);
    }

    @SneakyThrows
    private void prepareOriginalIndices(List<IndexConfigContext> indexConfigContexts) {
        assertEquals(6, indexConfigContexts.size());
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        createKnnIndex(derivedSourceEnabledContext.indexName, derivedSourceEnabledContext.settings, derivedSourceEnabledContext.mapping);
        createKnnIndex(derivedSourceDisabledContext.indexName, derivedSourceDisabledContext.settings, derivedSourceDisabledContext.mapping);
        derivedSourceEnabledContext.indexIngestor.accept(derivedSourceEnabledContext);
        derivedSourceDisabledContext.indexIngestor.accept(derivedSourceDisabledContext);
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            derivedSourceDisabledContext.indexName,
            derivedSourceEnabledContext.indexName
        );
    }

    @SneakyThrows
    private void testMerging(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 10);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    private void testUpdate(List<IndexConfigContext> indexConfigContexts) {
        // Random variables
        int docWithVectorUpdate = DOCS - 4;
        int docWithVectorRemoval = 1;
        int docWithVectorUpdateFromAPI = 2;
        int docWithUpdateByQuery = 7;

        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        Object updateVector = derivedSourceDisabledContext.updateVectorSupplier.apply(derivedSourceDisabledContext);

        // Update via POST /<index>/_doc/<docid>
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDoc(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithVectorUpdate), fieldName, updateVector);
        }

        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDoc(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithVectorUpdate), fieldName, updateVector);
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Sets the doc to an empty doc
        setDocToEmpty(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithVectorRemoval));
        setDocToEmpty(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithVectorRemoval));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Use update API
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDocWithUpdateAPI(
                originalIndexNameDerivedSourceEnabled,
                String.valueOf(docWithVectorUpdateFromAPI),
                fieldName,
                updateVector
            );
        }
        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDocWithUpdateAPI(
                originalIndexNameDerivedSourceDisabled,
                String.valueOf(docWithVectorUpdateFromAPI),
                fieldName,
                updateVector
            );
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Update by query
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDocByQuery(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithUpdateByQuery), fieldName, updateVector);
        }
        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDocByQuery(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithUpdateByQuery), fieldName, updateVector);
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    private void testSearch(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;

        // Default - all fields should be there
        validateSearch(originalIndexNameDerivedSourceEnabled, derivedSourceEnabledContext.docCount, true, null, null);

        // Default - no fields should be there
        validateSearch(originalIndexNameDerivedSourceEnabled, derivedSourceEnabledContext.docCount, false, null, null);

        // Exclude all vectors
        validateSearch(
            originalIndexNameDerivedSourceEnabled,
            derivedSourceEnabledContext.docCount,
            true,
            null,
            derivedSourceEnabledContext.vectorFieldNames
        );

        // Include all vectors
        validateSearch(
            originalIndexNameDerivedSourceEnabled,
            derivedSourceEnabledContext.docCount,
            true,
            derivedSourceEnabledContext.vectorFieldNames,
            null
        );
    }

    @SneakyThrows
    private void validateSearch(String indexName, int size, boolean isSourceEnabled, List<String> includes, List<String> excludes) {
        // TODO: We need to figure out a way to enhance validation
        QueryBuilder qb = new MatchAllQueryBuilder();
        Request request = new Request("POST", "/" + indexName + "/_search");

        request.addParameter("size", Integer.toString(size));
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field("query", qb);
        if (isSourceEnabled == false) {
            builder.field("_source", false);
        }
        if (includes != null) {
            builder.startObject("_source");
            builder.startArray("includes");
            for (String include : includes) {
                builder.value(include);
            }
            builder.endArray();
            builder.endObject();
        }
        if (excludes != null) {
            builder.startObject("_source");
            builder.startArray("excludes");
            for (String exclude : excludes) {
                builder.value(exclude);
            }
            builder.endArray();
            builder.endObject();
        }

        builder.endObject();
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        List<Object> hits = parseSearchResponseHits(responseBody);

        assertNotEquals(0, hits.size());
    }

    @SneakyThrows
    private void testDelete(List<IndexConfigContext> indexConfigContexts) {
        int docToDelete = 8;
        int docToDeleteByQuery = 11;

        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;

        // Delete by API
        deleteKnnDoc(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDelete));
        deleteKnnDoc(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDelete));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Delete by query
        deleteKnnDocByQuery(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDeleteByQuery));
        deleteKnnDocByQuery(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDeleteByQuery));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    private void testReindex(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        IndexConfigContext reindexFromEnabledToEnabledContext = indexConfigContexts.get(2);
        IndexConfigContext reindexFromEnabledToDisabledContext = indexConfigContexts.get(3);
        IndexConfigContext reindexFromDisabledToEnabledContext = indexConfigContexts.get(4);
        IndexConfigContext reindexFromDisabledToDisabledContext = indexConfigContexts.get(5);

        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        String reindexFromEnabledToEnabledIndexName = reindexFromEnabledToEnabledContext.indexName;
        String reindexFromEnabledToDisabledIndexName = reindexFromEnabledToDisabledContext.indexName;
        String reindexFromDisabledToEnabledIndexName = reindexFromDisabledToEnabledContext.indexName;
        String reindexFromDisabledToDisabledIndexName = reindexFromDisabledToDisabledContext.indexName;

        createKnnIndex(
            reindexFromEnabledToEnabledIndexName,
            reindexFromEnabledToEnabledContext.getSettings(),
            reindexFromEnabledToEnabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromEnabledToDisabledIndexName,
            reindexFromEnabledToDisabledContext.getSettings(),
            reindexFromEnabledToDisabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromDisabledToEnabledIndexName,
            reindexFromDisabledToEnabledContext.getSettings(),
            reindexFromDisabledToEnabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromDisabledToDisabledIndexName,
            reindexFromDisabledToDisabledContext.getSettings(),
            reindexFromDisabledToDisabledContext.getMapping()
        );
        refreshAllIndices();
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToDisabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToDisabledIndexName);

        // Need to forcemerge before comparison
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromEnabledToEnabledIndexName);
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        assertIndexBigger(reindexFromEnabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);
        assertIndexBigger(reindexFromDisabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromEnabledToEnabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromDisabledToEnabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromEnabledToDisabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromDisabledToDisabledIndexName
        );
    }

    @Builder
    @Data
    private static class IndexConfigContext {
        String indexName;
        List<String> vectorFieldNames;
        int dimension;
        Settings settings;
        String mapping;
        boolean isNested;
        int docCount;
        CheckedConsumer<IndexConfigContext, IOException> indexIngestor;
        Function<IndexConfigContext, Object> updateVectorSupplier;
    }

    @SneakyThrows
    private void assertIndexBigger(String expectedBiggerIndex, String expectedSmallerIndex) {
        if (isExhaustive()) {
            logger.info("Checking index bigger assertion because running in exhaustive mode");
            int expectedSmaller = indexSizeInBytes(expectedSmallerIndex);
            int expectedBigger = indexSizeInBytes(expectedBiggerIndex);
            assertTrue(
                "Expected smaller index " + expectedSmaller + " was bigger than the expected bigger index:" + expectedBigger,
                expectedSmaller < expectedBigger
            );
        } else {
            logger.info("Skipping index bigger assertion because not running in exhaustive mode");
        }
    }

    private void assertDocsMatch(int docCount, String index1, String index2) {
        for (int i = 0; i < docCount; i++) {
            assertDocMatches(i + 1, index1, index2);
        }
    }

    @SneakyThrows
    private void assertDocMatches(int docId, String index1, String index2) {
        Map<String, Object> response1 = getKnnDoc(index1, String.valueOf(docId));
        Map<String, Object> response2 = getKnnDoc(index2, String.valueOf(docId));
        assertEquals("Docs do not match: " + docId, response1, response2);
    }

    @SneakyThrows
    private String createVectorNonNestedMappings(final int dimension, String dataType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension);
        if (dataType != null) {
            builder.field(VECTOR_DATA_TYPE_FIELD, dataType);
        }
        builder.endObject().endObject().endObject();

        return builder.toString();
    }

    @SneakyThrows
    private String createVectorNestedMappings(final int dimension, String dataType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_NAME)
            .field(TYPE, "nested")
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension);
        if (dataType != null) {
            builder.field(VECTOR_DATA_TYPE_FIELD, dataType);
        }
        builder.endObject().endObject().endObject().endObject().endObject();
        return builder.toString();
    }
}
