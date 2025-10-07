/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.junit.Before;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.knn.Pair;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
    @ExpectRemoteBuildValidation
    public void testFlatFields() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("derivedit", true, true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @ExpectRemoteBuildValidation
    public void testMetaFields() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getIndexContextsWithMetaFields("derivedit", true, true);
        List<String> metaFields = List.of(ROUTING_FIELD, "_id", "_score");

        assertEquals("Expected 6 index contexts for meta fields test", 6, indexConfigContexts.size());
        prepareOriginalIndices(indexConfigContexts);

        List<Object> searchResults = testSearch(indexConfigContexts);
        assertFalse("Search results should not be empty", searchResults.isEmpty());

        for (int i = 0; i < searchResults.size(); i++) {
            Object searchResult = searchResults.get(i);
            assertNotNull("Search result at index " + i + " should not be null", searchResult);

            Map<String, Object> hits = (Map<String, Object>) searchResult;
            for (String metaField : metaFields) {
                assertTrue(String.format("Missing meta field '%s' in search result %d", metaField, i), hits.containsKey(metaField));
                assertNotNull(
                    String.format("Meta field '%s' value should not be null in search result %d", metaField, i),
                    hits.get(metaField)
                );
            }
        }
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testObjectField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getObjectIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testNestedField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testDerivedSource_whenSegrepLocal_thenDisabled() {
        // Set the data type input for float fields as byte. If derived source gets enabled, the original and derived
        // wont match because original will have source like [0, 1, 2] and derived will have [0.0, 1.0, 2.0]
        final List<Pair<String, Boolean>> indexPrefixToEnabled = List.of(
            new Pair<>("original-enable-", true),
            new Pair<>("original-disable-", false)
        );
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        long consistentRandomSeed = random().nextLong();
        for (Pair<String, Boolean> index : indexPrefixToEnabled) {
            Random random = new Random(consistentRandomSeed);
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName("deriveit", index.getFirst(), false))
                .derivedEnabled(index.getSecond())
                .random(random)
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
                                        .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.BYTE))
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
                                        .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.BYTE))
                                        .build(),
                                    DerivedSourceUtils.NestedFieldContext.builder()
                                        .fieldPath("nested_2.nested_3")
                                        .children(
                                            List.of(
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .fieldPath("nested_2.nested_3.test_vector")
                                                    .dimension(TEST_DIMENSION)
                                                    .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.BYTE))
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
                            .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.BYTE))
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
     * Tests that kNN handles bad documents the same when derived source is enabled and disabled.
     * @throws java.io.IOException
     */
    public void testDerivedSource_HandlesInvalidDocuments() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getCustomAnalyzerIndexContexts("derivedit", true, true);

        assertTrue(1 < indexConfigContexts.size());
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        createKnnIndex(
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.getSettings(),
            derivedSourceEnabledContext.getMapping()
        );
        createKnnIndex(
            derivedSourceDisabledContext.indexName,
            derivedSourceDisabledContext.getSettings(),
            derivedSourceDisabledContext.getMapping()
        );
        for (int i = 0; i < derivedSourceDisabledContext.docCount; i++) {
            String doc1 = derivedSourceEnabledContext.buildDoc();
            String doc2 = derivedSourceDisabledContext.buildDoc();
            assertEquals(doc1, doc2);
            boolean dsEnabledException = false;
            boolean dsDisabledException = false;
            try {
                addKnnDoc(derivedSourceEnabledContext.getIndexName(), String.valueOf(i + 1), doc1);
            } catch (ResponseException e) {
                assertTrue(e.getMessage().contains("number_format_exception"));
                dsEnabledException = true;
            }
            try {
                addKnnDoc(derivedSourceDisabledContext.getIndexName(), String.valueOf(i + 1), doc2);
            } catch (ResponseException e) {
                assertTrue(e.getMessage().contains("number_format_exception"));
                dsDisabledException = true;
            }
            assertEquals(dsEnabledException, dsDisabledException);
        }
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

    @SneakyThrows
    public void testBlockSettingIfKNNFalse() {
        String indexName = getIndexName("setting-blocked", "test", false);
        String fieldName = "test";
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
        expectThrows(
            ResponseException.class,
            () -> createKnnIndex(
                indexName,
                Settings.builder().put("index.knn", false).put("index.knn.derived_source.enabled", true).build(),
                mapping
            )
        );

        expectThrows(
            ResponseException.class,
            () -> createKnnIndex(indexName, Settings.builder().put("index.knn.derived_source.enabled", true).build(), mapping)
        );
    }
}
