/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.junit.Ignore;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality.
 *     //TODO: Dimensions:
 *     // 1. Data type
 *     // 2. Dimension
 *     // 3. Nested level
 *     // 4. Vectors per field
 *     // 5. Other fields
 *     // 6. Minimum number of values
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

    @SneakyThrows
    public void testFlatBaseCase() {
        String indexNameDerivedSourceEnabled = ("enabled-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String indexNameDerivedSourceDisabled = ("disabled-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        prepareFlatIndex(indexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        prepareFlatIndex(indexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        assertDocsMatch(DOCS, indexNameDerivedSourceEnabled, indexNameDerivedSourceDisabled);
        forceMergeKnnIndex(indexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(indexNameDerivedSourceDisabled, 10);
        refreshAllIndices();
        assertIndexBigger(indexNameDerivedSourceDisabled, indexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, indexNameDerivedSourceEnabled, indexNameDerivedSourceDisabled);
        refreshAllIndices();
        forceMergeKnnIndex(indexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(indexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(indexNameDerivedSourceDisabled, indexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, indexNameDerivedSourceEnabled, indexNameDerivedSourceDisabled);
    }

    @SneakyThrows
    public void testFlatReindex() {
        String originalIndexNameDerivedSourceEnabled = ("original-enable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String originalIndexNameDerivedSourceDisabled = ("original-disable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String reindexFromEnabledToEnabledIndexName = ("e2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String reindexFromEnabledToDisabledIndexName = ("e2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String reindexFromDisabledToEnabledIndexName = ("d2e-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String reindexFromDisabledToDisabledIndexName = ("d2d-" + getTestName() + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);

        prepareFlatIndex(originalIndexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        prepareFlatIndex(originalIndexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        createKnnIndex(reindexFromEnabledToEnabledIndexName, DERIVED_ENABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        createKnnIndex(reindexFromEnabledToDisabledIndexName, DERIVED_DISABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        createKnnIndex(reindexFromDisabledToEnabledIndexName, DERIVED_ENABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        createKnnIndex(reindexFromDisabledToDisabledIndexName, DERIVED_DISABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));

        refreshAllIndices();
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToDisabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToDisabledIndexName);
        refreshAllIndices();

        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromEnabledToEnabledIndexName);
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        assertIndexBigger(reindexFromEnabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);
        assertIndexBigger(reindexFromDisabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);

        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, reindexFromEnabledToEnabledIndexName);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, reindexFromEnabledToDisabledIndexName);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, reindexFromDisabledToDisabledIndexName);
    }

    @SneakyThrows
    public void testFlatDeletesAndUpdates() {
        String originalIndexNameDerivedSourceEnabled = ("original-enable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String originalIndexNameDerivedSourceDisabled = ("original-disable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        prepareFlatIndex(originalIndexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));
        prepareFlatIndex(originalIndexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, createVectorNonNestedMappings(TEST_DIMENSION));

        int docWithVectorUpdate = DOCS - 4;
        int docWithVectorRemoval = 1;
        int docWithVectorUpdateFromAPI = 2;
        int docWithUpdateByQuery = 7;
        int docToDelete = 8;
        int docToDeleteByQuery = 11;

        float[] updateVector = randomFloatVector(TEST_DIMENSION);
        updateKnnDoc(
            originalIndexNameDerivedSourceEnabled,
            String.valueOf(docWithVectorUpdate),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        updateKnnDoc(
            originalIndexNameDerivedSourceDisabled,
            String.valueOf(docWithVectorUpdate),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        setDocToEmpty(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithVectorRemoval));
        setDocToEmpty(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithVectorRemoval));
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        updateKnnDocWithUpdateAPI(
            originalIndexNameDerivedSourceEnabled,
            String.valueOf(docWithVectorUpdateFromAPI),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        updateKnnDocWithUpdateAPI(
            originalIndexNameDerivedSourceDisabled,
            String.valueOf(docWithVectorUpdateFromAPI),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        updateKnnDocByQuery(
            originalIndexNameDerivedSourceEnabled,
            String.valueOf(docWithUpdateByQuery),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        updateKnnDocByQuery(
            originalIndexNameDerivedSourceDisabled,
            String.valueOf(docWithUpdateByQuery),
            FIELD_NAME,
            Floats.asList(updateVector).toArray()
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        deleteKnnDoc(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDelete));
        deleteKnnDoc(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDelete));
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        deleteKnnDocByQuery(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDeleteByQuery));
        deleteKnnDocByQuery(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDeleteByQuery));
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
    }

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

        String originalIndexNameDerivedSourceEnabled = ("original-enable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String originalIndexNameDerivedSourceDisabled = ("original-disable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);

        createKnnIndex(originalIndexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, builder.toString());
        createKnnIndex(originalIndexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, builder.toString());
        bulkIngestRandomVectorsWithSkipsAndMultFields(
            originalIndexNameDerivedSourceEnabled,
            FIELD_NAME + "1",
            FIELD_NAME + "2",
            "text",
            DOCS,
            TEST_DIMENSION,
            0.1f
        );
        bulkIngestRandomVectorsWithSkipsAndMultFields(
            originalIndexNameDerivedSourceDisabled,
            FIELD_NAME + "1",
            FIELD_NAME + "2",
            "text",
            DOCS,
            TEST_DIMENSION,
            0.1f
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceEnabled, originalIndexNameDerivedSourceDisabled);
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 10);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceEnabled, originalIndexNameDerivedSourceDisabled);
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceEnabled, originalIndexNameDerivedSourceDisabled);

        int docWithVectorUpdate = DOCS - 4;
        int docWithUpdateByQuery = 7;

        float[] updateVector = randomFloatVector(TEST_DIMENSION);
        updateKnnDoc(
            originalIndexNameDerivedSourceEnabled,
            String.valueOf(docWithVectorUpdate),
            FIELD_NAME + "1",
            Floats.asList(updateVector).toArray()
        );
        updateKnnDoc(
            originalIndexNameDerivedSourceDisabled,
            String.valueOf(docWithVectorUpdate),
            FIELD_NAME + "1",
            Floats.asList(updateVector).toArray()
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        updateKnnDocByQuery(
            originalIndexNameDerivedSourceEnabled,
            String.valueOf(docWithUpdateByQuery),
            FIELD_NAME + "2",
            Floats.asList(updateVector).toArray()
        );
        updateKnnDocByQuery(
            originalIndexNameDerivedSourceDisabled,
            String.valueOf(docWithUpdateByQuery),
            FIELD_NAME + "3",
            Floats.asList(updateVector).toArray()
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
    }

    @Ignore
    @SneakyThrows
    public void testNestedSingleDocBasic() {
        // For basic tests, we will have 0-5 nested documents per document
        String originalIndexNameDerivedSourceEnabled = ("original-enable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);
        String originalIndexNameDerivedSourceDisabled = ("original-disable-" + randomAlphaOfLength(6)).toLowerCase(Locale.ROOT);

        createKnnIndex(originalIndexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, createVectorNestedMappings(TEST_DIMENSION));
        createKnnIndex(originalIndexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, createVectorNestedMappings(TEST_DIMENSION));

        bulkIngestRandomVectorsWithSkipsAndNested(
            originalIndexNameDerivedSourceEnabled,
            NESTED_NAME + "." + FIELD_NAME,
            NESTED_NAME + "." + "text",
            DOCS,
            TEST_DIMENSION,
            0.1f
        );
        bulkIngestRandomVectorsWithSkipsAndNested(
            originalIndexNameDerivedSourceDisabled,
            NESTED_NAME + "." + FIELD_NAME,
            NESTED_NAME + "." + "text",
            DOCS,
            TEST_DIMENSION,
            0.1f
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceEnabled, originalIndexNameDerivedSourceDisabled);
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 10);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceEnabled, originalIndexNameDerivedSourceDisabled);
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
    }

    @Ignore
    @SneakyThrows
    public void testNestedMultiDocBasic() {
        String originalIndexNameDerivedSourceEnabled = ("original-enable-" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT)); // "test");
        // /*randomAlphaOfLength(4)).toLowerCase(Locale.ROOT)*/;
        String originalIndexNameDerivedSourceDisabled = ("original-disable-" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT)); // + ;
        // //"test");
        // /*randomAlphaOfLength(4)).toLowerCase(Locale.ROOT)*/;

        createKnnIndex(originalIndexNameDerivedSourceEnabled, DERIVED_ENABLED_SETTINGS, createVectorNestedMappings(TEST_DIMENSION));
        createKnnIndex(originalIndexNameDerivedSourceDisabled, DERIVED_DISABLED_SETTINGS, createVectorNestedMappings(TEST_DIMENSION));

        bulkIngestRandomVectorsWithSkipsAndNestedMultiDoc(
            originalIndexNameDerivedSourceEnabled,
            NESTED_NAME + "." + FIELD_NAME,
            NESTED_NAME + "." + "text",
            DOCS,
            TEST_DIMENSION,
            0.1f,
            5
        );
        bulkIngestRandomVectorsWithSkipsAndNestedMultiDoc(
            originalIndexNameDerivedSourceDisabled,
            NESTED_NAME + "." + FIELD_NAME,
            NESTED_NAME + "." + "text",
            DOCS,
            TEST_DIMENSION,
            0.1f,
            5
        );
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 10);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertDocsMatch(DOCS, originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
    }

    // public void testNestedReindex() {
    //
    // }
    //
    // public void testNestedUpdateAndDelete() {
    //
    // }
    //
    // public void testMultiNestedFields() {
    // // TODO
    // }
    //
    // public void testMixedNestedAndFlatFields() {
    // // TODO
    // }
    //
    // public void testFLSSupport() {
    // // TODO: Security only - need to figure out how to configure this one better
    // }
    //
    // public void testNullSet() {
    // // TODO: we know this breaks
    // }

    @SneakyThrows
    private void assertIndexBigger(String expectedBiggerIndex, String expectedSmallerIndex) {
        assertTrue(indexSizeInBytes(expectedSmallerIndex) < indexSizeInBytes(expectedBiggerIndex));
    }

    @SneakyThrows
    private void prepareFlatIndex(String indexName, Settings settings, String mapping) {
        createKnnIndex(indexName, settings, mapping);
        bulkIngestRandomVectorsWithSkips(indexName, FIELD_NAME, DOCS, TEST_DIMENSION, 0.1f);
        refreshAllIndices();
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
    private String createVectorNonNestedMappings(final int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }

    @SneakyThrows
    private String createVectorNestedMappings(final int dimension) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_NAME)
            .field(TYPE, "nested")
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }
}
