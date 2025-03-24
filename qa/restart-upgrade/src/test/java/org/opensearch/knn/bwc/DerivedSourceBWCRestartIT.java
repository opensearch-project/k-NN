/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.BWC_VERSION;
import static org.opensearch.knn.TestUtils.CLIENT_TIMEOUT_VALUE;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.RESTART_UPGRADE_OLD_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

public class DerivedSourceBWCRestartIT extends DerivedSourceTestCase {

    public void testFlat_indexAndForceMergeOnOld_injectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getFlatIndexContexts();
        testIndexAndForceMergeOnOld_injectOnNew(indexConfigContexts);
    }

    public void testObject_indexAndForceMergeOnOld_injectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getObjectIndexContexts();
        testIndexAndForceMergeOnOld_injectOnNew(indexConfigContexts);
    }

    public void testNested_indexAndForceMergeOnOld_injectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getNestedIndexContexts();
        testIndexAndForceMergeOnOld_injectOnNew(indexConfigContexts);
    }

    private void testIndexAndForceMergeOnOld_injectOnNew(List<IndexConfigContext> indexConfigContexts) throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
            testMerging(indexConfigContexts);
            // Update. Skipping update tests for nested docs for now. Will add in the future.
            if (indexConfigContexts.get(0).isNested() == false) {
                testUpdate(indexConfigContexts);
            }

            // Delete
            testDelete(indexConfigContexts);
            assertDocsMatch(indexConfigContexts);
        } else {
            assertDocsMatch(indexConfigContexts);
            // Search
            testSearch(indexConfigContexts);

            // Reindex
            testReindex(indexConfigContexts);
        }
    }

    public void testFlat_indexOnOld_forceMergeAndInjectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getFlatIndexContexts();
        testIndexOnOld_forceMergeAndInjectOnNew(indexConfigContexts);
    }

    public void testObject_indexOnOld_forceMergeAndInjectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getObjectIndexContexts();
        testIndexOnOld_forceMergeAndInjectOnNew(indexConfigContexts);
    }

    public void testNested_indexOnOld_forceMergeAndInjectOnNew() throws IOException {
        List<IndexConfigContext> indexConfigContexts = getNestedIndexContexts();
        testIndexOnOld_forceMergeAndInjectOnNew(indexConfigContexts);
    }

    private void testIndexOnOld_forceMergeAndInjectOnNew(List<IndexConfigContext> indexConfigContexts) throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
        } else {
            assertDocsMatch(indexConfigContexts);
            testMerging(indexConfigContexts);
            assertDocsMatch(indexConfigContexts);
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

    private List<IndexConfigContext> getFlatIndexContexts() {
        String mapping = createVectorNonNestedMappings(TEST_DIMENSION, null);
        return List.of(
            IndexConfigContext.builder()
                .indexName(("knn-bwc-original-enable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-original-disable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2e-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-d2e-" + getTestName()).toLowerCase(Locale.ROOT))
                .vectorFieldNames(List.of(FIELD_NAME))
                .dimension(TEST_DIMENSION)
                .settings(DERIVED_ENABLED_SETTINGS)
                .mapping(mapping)
                .isNested(false)
                .docCount(DOCS)
                .indexIngestor(context -> {}) // noop for reindex
                .build(),
            IndexConfigContext.builder()
                .indexName(("knn-bwc-d2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
    }

    private List<IndexConfigContext> getObjectIndexContexts() throws IOException {
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

        return List.of(
            IndexConfigContext.builder()
                .indexName(("knn-bwc-original-enable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-original-disable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2e-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-d2e-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-d2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
    }

    private List<IndexConfigContext> getNestedIndexContexts() {
        String nestedMapping = createVectorNestedMappings(TEST_DIMENSION, null);
        return List.of(
            IndexConfigContext.builder()
                .indexName(("knn-bwc-original-enable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-original-disable-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2e-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-e2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-d2e-" + getTestName()).toLowerCase(Locale.ROOT))
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
                .indexName(("knn-bwc-d2d-" + getTestName()).toLowerCase(Locale.ROOT))
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
    }

    @Override
    protected final boolean preserveIndicesUponCompletion() {
        return true;
    }

    @Override
    protected final boolean preserveReposUponCompletion() {
        return true;
    }

    @Override
    protected boolean preserveTemplatesUponCompletion() {
        return true;
    }

    @Override
    protected final Settings restClientSettings() {
        return Settings.builder()
            .put(super.restClientSettings())
            // increase the timeout here to 90 seconds to handle long waits for a green
            // cluster health. the waits for green need to be longer than a minute to
            // account for delayed shards
            .put(OpenSearchRestTestCase.CLIENT_SOCKET_TIMEOUT, CLIENT_TIMEOUT_VALUE)
            .build();
    }

    protected static final boolean isRunningAgainstOldCluster() {
        return Boolean.parseBoolean(System.getProperty(RESTART_UPGRADE_OLD_CLUSTER));
    }

    @Override
    protected final Optional<String> getBWCVersion() {
        return Optional.ofNullable(System.getProperty(BWC_VERSION, null));
    }

}
