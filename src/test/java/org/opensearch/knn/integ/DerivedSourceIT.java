/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.knn.Pair;
import org.opensearch.knn.common.KNNConstants;
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
import static org.opensearch.knn.common.KNNConstants.DIMENSION;

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
     * Tests that bulk indexing with dynamic templates works correctly when derived source is enabled.
     * This is a regression test for https://github.com/opensearch-project/k-NN/issues/3012
     *
     * The bug: When bulk indexing documents with dynamic templates that create different field
     * mappings per document, only the
     * first document's vector was correctly reconstructed. Subsequent documents returned the
     * mask value (1) instead of the actual vector.
     *
     * Root cause: Segment attributes were written at segment creation time (in fieldsWriter()),
     * when only the first document's dynamic mapping existed. The fix moves attribute writing
     * to finish(), when all documents have been parsed and all mappings exist.
     */
    @SneakyThrows
    public void testDerivedSource_withDynamicTemplates_andBulkIndexing() {
        String indexName = "test-derived-dynamic-template";
        int dimension = 3;

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.derived_source.enabled", true)
            .build();

        XContentBuilder mappingsBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startArray("dynamic_templates")
            .startObject()
            .startObject("knn_vector_template")
            .field("path_match", "similar_products_vector.*.clip_vit_base_patch32")
            .startObject("mapping")
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("engine", "faiss")
            .field("space_type", "l2")
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endArray()
            .endObject();

        createKnnIndex(indexName, settings, mappingsBuilder.toString());

        StringBuilder bulkRequestBody = new StringBuilder();

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc1\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1001\": {\"clip_vit_base_patch32\": [1.0, 1.0, 1.0]}}}\n");

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc2\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1002\": {\"clip_vit_base_patch32\": [2.0, 2.0, 2.0]}}}\n");

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc3\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1003\": {\"clip_vit_base_patch32\": [3.0, 3.0, 3.0]}}}\n");

        Request bulkRequest = new Request("POST", "/_bulk");
        bulkRequest.setJsonEntity(bulkRequestBody.toString());
        Response bulkResponse = client().performRequest(bulkRequest);
        assertEquals(RestStatus.OK.getStatus(), bulkResponse.getStatusLine().getStatusCode());

        refreshIndex(indexName);

        Map<String, Object> doc1Source = getKnnDoc(indexName, "doc1");
        Map<String, Object> doc2Source = getKnnDoc(indexName, "doc2");
        Map<String, Object> doc3Source = getKnnDoc(indexName, "doc3");

        List<Float> retrievedVector1 = extractVector(doc1Source, "similar_products_vector", "key_1001", "clip_vit_base_patch32");
        List<Float> retrievedVector2 = extractVector(doc2Source, "similar_products_vector", "key_1002", "clip_vit_base_patch32");
        List<Float> retrievedVector3 = extractVector(doc3Source, "similar_products_vector", "key_1003", "clip_vit_base_patch32");

        assertNotNull("Vector 1 should not be null - got mask value instead of array", retrievedVector1);
        assertNotNull("Vector 2 should not be null - got mask value instead of array", retrievedVector2);
        assertNotNull("Vector 3 should not be null - got mask value instead of array", retrievedVector3);

        assertEquals("Vector 1 should have correct dimension", dimension, retrievedVector1.size());
        assertEquals("Vector 2 should have correct dimension", dimension, retrievedVector2.size());
        assertEquals("Vector 3 should have correct dimension", dimension, retrievedVector3.size());

        deleteKNNIndex(indexName);
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

    @SneakyThrows
    public void testSourceFiltering_withVariousIncludeExcludeCombinations() {
        String indexName = getIndexName("source-filtering", "combinations", false);
        String VECTOR_FIELD_1 = "test_vector";
        String VECTOR_FIELD_2 = "temp_vector";
        String VECTOR_FIELD_3 = "user_vector";
        String TEXT_FIELD = "description";
        int DIMENSION = 3;

        // Create index with multiple vector fields and a text field
        XContentBuilder mappingBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(VECTOR_FIELD_3)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexName,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingBuilder.toString()
        );

        // Index a document with all fields
        XContentBuilder docBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .array(VECTOR_FIELD_3, 7.0f, 8.0f, 9.0f)
            .field(TEXT_FIELD, "test description")
            .endObject();
        addKnnDoc(indexName, "1", docBuilder.toString());

        refreshIndex(indexName);

        // Test 1: No filtering - all fields returned
        assertSourceFiltering(
            indexName,
            null,  // includes
            null,  // excludes
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },  // expected present
            new String[] {}  // expected absent
        );

        // Test 2: Only includes - only specified fields returned
        assertSourceFiltering(
            indexName,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            null,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3 }
        );

        // Test 3: Only excludes - all except specified fields returned
        assertSourceFiltering(
            indexName,
            null,
            new String[] { VECTOR_FIELD_1 },
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] { VECTOR_FIELD_1 }
        );

        // Test 4: Both includes and excludes - excludes override includes
        assertSourceFiltering(
            indexName,
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2 },
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3 }
        );

        // Test 5: Wildcard includes - only matching fields returned
        assertSourceFiltering(
            indexName,
            new String[] { "t*" },  // matches test_vector, temp_vector
            null,
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2 },
            new String[] { VECTOR_FIELD_3, TEXT_FIELD }
        );

        // Test 6: Wildcard excludes - all except matching fields returned
        assertSourceFiltering(
            indexName,
            null,
            new String[] { "t*" },  // excludes test_vector, temp_vector
            new String[] { VECTOR_FIELD_3, TEXT_FIELD },
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2 }
        );

        // Test 7: Wildcard includes with specific excludes
        assertSourceFiltering(
            indexName,
            new String[] { "t*", VECTOR_FIELD_3 },  // includes test_vector, temp_vector, user_vector
            new String[] { VECTOR_FIELD_1 },  // excludes test_vector
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3 },
            new String[] { VECTOR_FIELD_1, TEXT_FIELD }
        );

        // Test 8: Empty includes array - all fields returned (no filtering)
        assertSourceFiltering(
            indexName,
            new String[] {},
            null,
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] {}
        );

        // Test 9: Empty excludes array - all fields returned (no filtering)
        assertSourceFiltering(
            indexName,
            null,
            new String[] {},
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] {}
        );
    }

    @SuppressWarnings("unchecked")
    private List<Float> extractVector(Map<String, Object> source, String... path) {
        Object current = source;
        for (String key : path) {
            if (current instanceof Map) {
                current = ((Map<String, Object>) current).get(key);
            } else {
                return null;
            }
        }
        if (current instanceof List) {
            return (List<Float>) current;
        }
        return null;
    }

    @SneakyThrows
    public void testDerivedSource_withMappingLevelSourceFiltering() {
        String VECTOR_FIELD_1 = "included_vector";
        String VECTOR_FIELD_2 = "excluded_vector";
        String TEXT_FIELD = "title";
        int dimension = 3;

        // Test 1: Index with _source.includes - only included_vector should be in source
        String indexWithIncludes = getIndexName("source-mapping", "includes", false);
        XContentBuilder mappingWithIncludes = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("includes", VECTOR_FIELD_1, TEXT_FIELD)
            .endObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexWithIncludes,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingWithIncludes.toString()
        );

        XContentBuilder docBuilder1 = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .field(TEXT_FIELD, "test document")
            .endObject();
        addKnnDoc(indexWithIncludes, "1", docBuilder1.toString());
        refreshIndex(indexWithIncludes);

        // Verify - included_vector should be present (derived), excluded_vector should be absent
        assertMappingLevelSourceFiltering(indexWithIncludes, new String[] { VECTOR_FIELD_1, TEXT_FIELD }, new String[] { VECTOR_FIELD_2 });

        // Test 2: Index with _source.excludes - excluded_vector should not be in source
        String indexWithExcludes = getIndexName("source-mapping", "excludes", false);
        XContentBuilder mappingWithExcludes = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("excludes", VECTOR_FIELD_2)
            .endObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexWithExcludes,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingWithExcludes.toString()
        );

        XContentBuilder docBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .field(TEXT_FIELD, "test document")
            .endObject();
        addKnnDoc(indexWithExcludes, "1", docBuilder2.toString());
        refreshIndex(indexWithExcludes);

        // Verify
        assertMappingLevelSourceFiltering(indexWithExcludes, new String[] { VECTOR_FIELD_1, TEXT_FIELD }, new String[] { VECTOR_FIELD_2 });

        // Test 3: Index with both _source.includes and _source.excludes - excludes override includes
        String indexWithBoth = getIndexName("source-mapping", "both", false);
        XContentBuilder mappingWithBoth = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("includes", VECTOR_FIELD_1, VECTOR_FIELD_2, TEXT_FIELD)
            .array("excludes", VECTOR_FIELD_2)
            .endObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexWithBoth,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingWithBoth.toString()
        );

        XContentBuilder docBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .field(TEXT_FIELD, "test document")
            .endObject();
        addKnnDoc(indexWithBoth, "1", docBuilder3.toString());
        refreshIndex(indexWithBoth);

        // Verify - excludes should override includes
        assertMappingLevelSourceFiltering(indexWithBoth, new String[] { VECTOR_FIELD_1, TEXT_FIELD }, new String[] { VECTOR_FIELD_2 });

        // Test 4: Wildcard includes
        String indexWithWildcardIncludes = getIndexName("source-mapping", "wildcard-includes", false);
        XContentBuilder mappingWithWildcardIncludes = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("includes", "included_*", TEXT_FIELD)
            .endObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexWithWildcardIncludes,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingWithWildcardIncludes.toString()
        );

        XContentBuilder docBuilder4 = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .field(TEXT_FIELD, "test document")
            .endObject();
        addKnnDoc(indexWithWildcardIncludes, "1", docBuilder4.toString());
        refreshIndex(indexWithWildcardIncludes);

        // Verify - only included_vector matches wildcard
        assertMappingLevelSourceFiltering(
            indexWithWildcardIncludes,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2 }
        );

        // Test 5: Wildcard excludes
        String indexWithWildcardExcludes = getIndexName("source-mapping", "wildcard-excludes", false);
        XContentBuilder mappingWithWildcardExcludes = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("excludes", "excluded_*")
            .endObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexWithWildcardExcludes,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingWithWildcardExcludes.toString()
        );

        XContentBuilder docBuilder5 = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .field(TEXT_FIELD, "test document")
            .endObject();
        addKnnDoc(indexWithWildcardExcludes, "1", docBuilder5.toString());
        refreshIndex(indexWithWildcardExcludes);

        // Verify - excluded_vector matches wildcard
        assertMappingLevelSourceFiltering(
            indexWithWildcardExcludes,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2 }
        );
    }

    @SneakyThrows
    private void assertMappingLevelSourceFiltering(String indexName, String[] expectedPresent, String[] expectedAbsent) {
        XContentBuilder searchBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject();

        Request searchRequest = new Request("POST", "/" + indexName + "/_search");
        searchRequest.setJsonEntity(searchBuilder.toString());
        Response response = client().performRequest(searchRequest);

        Map<String, Object> responseMap = entityAsMap(response);
        Map<String, Object> hits = (Map<String, Object>) responseMap.get("hits");
        List<Map<String, Object>> hitsList = (List<Map<String, Object>>) hits.get("hits");

        assertEquals("Expected 1 hit", 1, hitsList.size());
        Map<String, Object> source = (Map<String, Object>) hitsList.get(0).get("_source");

        for (String field : expectedPresent) {
            assertTrue(
                String.format(Locale.ROOT, "Field '%s' should be present in _source for index '%s'", field, indexName),
                source.containsKey(field)
            );
        }

        for (String field : expectedAbsent) {
            assertFalse(
                String.format(Locale.ROOT, "Field '%s' should be absent from _source for index '%s'", field, indexName),
                source.containsKey(field)
            );
        }
    }

}
