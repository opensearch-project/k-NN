/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;

import java.util.List;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality. Ignoring tests for now as feature is experimental.
 */
public class DerivedSourceIT extends DerivedSourceTestCase {

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

}
