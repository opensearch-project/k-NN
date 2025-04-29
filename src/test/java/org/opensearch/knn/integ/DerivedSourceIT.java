/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.junit.Before;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.knn.Pair;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.opensearch.knn.DerivedSourceUtils.DERIVED_ENABLED_WITH_SEGREP_SETTINGS;
import static org.opensearch.knn.DerivedSourceUtils.TEST_DIMENSION;
import static org.opensearch.knn.DerivedSourceUtils.randomVectorSupplier;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality. Ignoring tests for now as feature is experimental.
 */
public class DerivedSourceIT extends DerivedSourceTestCase {

    private final String snapshot = "snapshot-test";
    private final String repository = "repo";

    @Before
    @SneakyThrows
    public void setUp() {
        super.setUp();
        final String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repository, "fs", true, repoSettings);
    }

    @SneakyThrows
    public void testFlatFields() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testObjectField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getObjectIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testNestedField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testDerivedSource_whenSegrepLocal_thenDisabled() {
        // Set the data type input for float fields as byte. If derived source gets enabled, the original and derived
        // wont match because original will have source like [0, 1, 2] and derived will have [0.0, 1.0, 2.0]
        final List<Pair<String, Boolean>> indexPrefixToEnabled = List.of(
            new Pair<>("original-enable-", true),
            new Pair<>("original-disable-", false)
        );
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        for (Pair<String, Boolean> index : indexPrefixToEnabled) {
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName("deriveit", index.getFirst(), false))
                .derivedEnabled(index.getSecond())
                .random(new Random(1))
                .settings(index.getSecond() ? DERIVED_ENABLED_WITH_SEGREP_SETTINGS : null)
                .fields(
                    List.of(
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_1")
                            .children(
                                List.of(
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .fieldPath("nested_1.test_vector")
                                        .dimension(TEST_DIMENSION)
                                        .valueSupplier(randomVectorSupplier(new Random(0), TEST_DIMENSION, VectorDataType.BYTE))
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_2")
                            .children(
                                List.of(
                                    DerivedSourceUtils.TextFieldType.builder().fieldPath("nested_2.test-text").build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .fieldPath("nested_2.test_vector")
                                        .dimension(TEST_DIMENSION)
                                        .valueSupplier(randomVectorSupplier(new Random(0), TEST_DIMENSION, VectorDataType.BYTE))
                                        .build(),
                                    DerivedSourceUtils.NestedFieldContext.builder()
                                        .fieldPath("nested_2.nested_3")
                                        .children(
                                            List.of(
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .fieldPath("nested_2.nested_3.test_vector")
                                                    .dimension(TEST_DIMENSION)
                                                    .valueSupplier(randomVectorSupplier(new Random(0), TEST_DIMENSION, VectorDataType.BYTE))
                                                    .build(),
                                                DerivedSourceUtils.IntFieldType.builder().fieldPath("nested_2.nested_3.test-int").build()
                                            )
                                        )
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(TEST_DIMENSION)
                            .valueSupplier(randomVectorSupplier(new Random(0), TEST_DIMENSION, VectorDataType.BYTE))
                            .fieldPath("test_vector")
                            .build(),
                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").build()
                    )
                )
                .build();
            indexConfigContext.init();
            indexConfigContexts.add(indexConfigContext);
        }

        prepareOriginalIndices(indexConfigContexts);
    }

    /**
     * Single method for running end to end tests for different index configurations for derived source. In general,
     * flow of operations are
     *
     * @param indexConfigContexts {@link DerivedSourceUtils.IndexConfigContext}
     */
    @SneakyThrows
    private void testDerivedSourceE2E(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        assertEquals(6, indexConfigContexts.size());

        // Prepare the indices by creating them and ingesting data into them
        prepareOriginalIndices(indexConfigContexts);

        // Merging
        testMerging(indexConfigContexts);

        // Update. Skipping update tests for nested docs for now. Will add in the future.
        testUpdate(indexConfigContexts);

        // Delete
        testDelete(indexConfigContexts);

        // Search
        testSearch(indexConfigContexts);

        // Reindex
        testReindex(indexConfigContexts);

        // Snapshot restore
        testSnapshotRestore(repository, snapshot + getTestName().toLowerCase(Locale.ROOT), indexConfigContexts);
    }

    @SneakyThrows
    public void testDefaultSetting() {
        String indexName = getIndexName("defaults", "test", false);
        String fieldName = "test";
        String indexNameDisabled = "disabled";
        int dimension = 16;
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createKnnIndex(indexName, mapping);
        validateDerivedSetting(indexName, true);
        createIndex(indexNameDisabled, Settings.builder().build());
        validateDerivedSetting(indexNameDisabled, false);
    }
}
