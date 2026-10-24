/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.common.KNNConstants;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.TestUtils.BWC_VERSION;
import static org.opensearch.knn.TestUtils.CLIENT_TIMEOUT_VALUE;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.RESTART_UPGRADE_OLD_CLUSTER;

public class DerivedSourceBWCRestartIT extends DerivedSourceTestCase {

    public DerivedSourceBWCRestartIT(CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @ParametersFactory(argumentFormatting = "compression:%1$s")
    public static Collection<Object[]> compressionParameters() {
        return List.<Object[]>of(new Object[] { CompressionTestConfig.X1 });
    }

    public void testFlat_indexAndForceMergeOnOld_injectOnNew() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("knn-bwc", false, false);
        testIndexAndForceMergeOnOld_injectOnNew(indexConfigContexts);
    }

    public void testFlat_indexOnOld_forceMergeAndInjectOnNew() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("knn-bwc", false, false);
        testIndexOnOld_forceMergeAndInjectOnNew(indexConfigContexts);
    }

    private void testIndexAndForceMergeOnOld_injectOnNew(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts)
        throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
            testMerging(indexConfigContexts);

            // Delete
            testDelete(indexConfigContexts);
        } else {
            // Search
            testSearch(indexConfigContexts);

            // Reindex
            testReindex(indexConfigContexts);
        }
    }

    private void testIndexOnOld_forceMergeAndInjectOnNew(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts)
        throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            prepareOriginalIndices(indexConfigContexts);
        } else {
            testMerging(indexConfigContexts);

            // Delete
            testDelete(indexConfigContexts);
            // Search
            testSearch(indexConfigContexts);

            // Reindex
            testReindex(indexConfigContexts);
        }
    }

    public void testOldSettingPreservedOnUpgrade() throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String indexName = getIndexName("knn-bwc", "defaults-", false);
        if (isRunningAgainstOldCluster()) {
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
            createKnnIndex(indexName, mapping);
            validateDerivedSetting(indexName, false);
        } else {
            validateDerivedSetting(indexName, false);
        }
    }

    public void testDerivedEnabledSettingPreservedOnUpgrade() throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String indexName = getIndexName("knn-bwc", "derived-enabled-", false);
        if (isRunningAgainstOldCluster()) {
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
            Settings settings = Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build();
            createKnnIndex(indexName, settings, mapping);
            validateDerivedSetting(indexName, true);
        } else {
            validateDerivedSetting(indexName, true);
        }
    }

    public void testMixedCaseDerivedSourceField_indexOnOld_reconstructOnNew() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String indexName = getIndexName("knn-bwc", "mixed-case-derived-source-", false);
        int dimension = 3;

        if (isRunningAgainstOldCluster()) {
            Settings settings = Settings.builder()
                .put("number_of_shards", 1)
                .put("number_of_replicas", 0)
                .put("index.knn", true)
                .put("index.knn.derived_source.enabled", true)
                .build();

            createKnnIndex(indexName, settings, createMixedCaseVectorMapping(dimension));
            addKnnDoc(indexName, "1", createMixedCaseVectorDoc());
            refreshIndex(indexName);
            flushIndex(indexName);
        } else {
            refreshIndex(indexName);
            List<?> retrievedVector = extractVector(getKnnDoc(indexName, "1"), "vectorSearch", "nameVector");

            assertNotNull("Mixed-case vector field should be reconstructed from old lowercase segment metadata", retrievedVector);
            assertEquals(dimension, retrievedVector.size());
            assertEquals(1.0f, ((Number) retrievedVector.get(0)).floatValue(), 0.0f);
            assertEquals(2.0f, ((Number) retrievedVector.get(1)).floatValue(), 0.0f);
            assertEquals(3.0f, ((Number) retrievedVector.get(2)).floatValue(), 0.0f);

            deleteKNNIndex(indexName);
        }
    }

    private String createMixedCaseVectorMapping(int dimension) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("vectorSearch")
            .startObject("properties")
            .startObject("nameVector")
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("engine", "lucene")
            .field("space_type", "l2")
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder.toString();
    }

    private String createMixedCaseVectorDoc() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("vectorSearch")
            .array("nameVector", 1.0f, 2.0f, 3.0f)
            .endObject()
            .endObject();
        return builder.toString();
    }

    // BWC regression test for the derived-source _source-bloat-on-merge fix (#3465, issue #3316). We build the
    // same index (derived source + _source.exclude) with the same documents on each cluster and compare sizes:
    // - oldBuilt: created, indexed, and force-merged on the OLD (pre-fix) cluster. Its force-merge re-injects
    // the excluded vector back into _source, so it is bloated. It is kept (not deleted) so it survives the
    // restart as the "before" reference.
    // - newBuilt: a separate index created, indexed, and force-merged on the UPGRADED (fixed) cluster with the
    // same documents. The fix keeps the vector out of _source on merge.
    // The new-cluster index must therefore be strictly smaller than the pre-fix one; both are deleted at the end.
    // This only holds when the old cluster predates the fix, so build.gradle excludes this test for BWC >= 3.8.
    public void testDerivedSourceExcludeMergeBloatBefore3_8_0() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        String vectorField = "test_vector";
        String excludedKeywordField = "label";
        int dimension = 128;

        String oldBuilt = getIndexName("knn-bwc", "old-exclude-", false);
        String newBuilt = getIndexName("knn-bwc", "new-exclude-", false);

        if (isRunningAgainstOldCluster()) {
            // Build and force-merge on the pre-fix cluster: the merge re-injects vectors into _source (#3316),
            // so this index is bloated. It is kept (not deleted here) and survives the restart as the reference.
            createSourceExcludeIndex(oldBuilt, vectorField, excludedKeywordField, dimension, true, true);
            indexSourceExcludeDocs(List.of(oldBuilt), vectorField, excludedKeywordField, dimension, 0);
            forceMergeKnnIndex(oldBuilt, 1);
        } else {
            try {
                // Create a new index and index the same documents on the upgraded (fixed) cluster, then merge.
                createSourceExcludeIndex(newBuilt, vectorField, excludedKeywordField, dimension, true, true);
                indexSourceExcludeDocs(List.of(newBuilt), vectorField, excludedKeywordField, dimension, 0);
                forceMergeKnnIndex(newBuilt, 1);

                // Correctness (the #2472 guard): a search fetching 10+ adjacent documents still reconstructs the
                // full vector into _source, and the excluded field is absent.
                assertVectorPresentAndFieldExcluded(newBuilt, vectorField, excludedKeywordField, dimension, 20);

                // kNN search must still work after the upgrade + force-merge - the fix only changes what is
                // stored in _source, not what the vector index returns.
                float[] queryVector = new float[dimension];
                Random queryRandom = new Random(4321);
                for (int d = 0; d < dimension; d++) {
                    queryVector[d] = queryRandom.nextFloat();
                }
                int k = 10;
                assertEquals(
                    "Expected k results from the new-cluster index",
                    k,
                    knnSearchDocIds(newBuilt, vectorField, queryVector, k).size()
                );

                int oldBuiltSize = indexSizeInBytes(oldBuilt);
                int newBuiltSize = indexSizeInBytes(newBuilt);

                // Same mapping and same documents, but the fixed code does not re-inject vectors into _source on
                // merge, so the new-cluster index must be strictly smaller than the pre-fix (bloated) one. The
                // vector index files are identical across both, so the difference is exactly the un-bloated _source.
                assertTrue(
                    String.format(
                        Locale.ROOT,
                        "New-cluster (fixed) index (%d bytes) should be smaller than the old-cluster (pre-fix) index "
                            + "(%d bytes) for the same documents, since the fix keeps vectors out of _source on merge",
                        newBuiltSize,
                        oldBuiltSize
                    ),
                    newBuiltSize < oldBuiltSize
                );
            } finally {
                // BWC tests preserve indices across the restart (preserveIndicesUponCompletion() == true), so both
                // are cleaned up explicitly at the end. Best-effort deletes so a failed assertion above does not
                // leak the indices into later tests in this suite.
                deleteIndicesQuietly(oldBuilt, newBuilt);
            }
        }
    }

    /**
     * Best-effort delete of the given indices, used on cleanup paths where a delete failure (for example a
     * missing index) must not mask the real test failure.
     */
    private void deleteIndicesQuietly(String... indices) {
        for (String index : indices) {
            try {
                deleteKNNIndex(index);
            } catch (Exception e) {
                logger.warn("Failed to delete index [" + index + "] during cleanup", e);
            }
        }
    }

    // Each phase writes NUM_BATCHES segments (one refresh per batch) so the subsequent force-merge has
    // >1 segment to combine. The #3316 bug only manifests when a segment merge actually runs.
    private static final int NUM_BATCHES = 5;
    private static final int DOCS_PER_BATCH = 20;

    private void indexSourceExcludeDocs(
        List<String> indices,
        String vectorField,
        String excludedKeywordField,
        int dimension,
        int startIdInclusive
    ) throws IOException {
        int nextId = startIdInclusive;
        for (int batch = 0; batch < NUM_BATCHES; batch++) {
            indexSourceExcludeBatch(indices, vectorField, excludedKeywordField, dimension, nextId, nextId + DOCS_PER_BATCH - 1);
            nextId += DOCS_PER_BATCH;
        }
    }

    // Indexes docs [startIdInclusive, endIdInclusive] into every index and refreshes once, producing a
    // single new segment per index (addKnnDoc does not refresh per doc).
    private void indexSourceExcludeBatch(
        List<String> indices,
        String vectorField,
        String excludedKeywordField,
        int dimension,
        int startIdInclusive,
        int endIdInclusive
    ) throws IOException {
        // Seed by start id so the batch generates distinct-but-reproducible vectors.
        Random random = new Random(startIdInclusive);
        for (int i = startIdInclusive; i <= endIdInclusive; i++) {
            float[] vector = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                vector[d] = random.nextFloat();
            }
            String docId = String.valueOf(i);
            String doc = XContentFactory.jsonBuilder()
                .startObject()
                .field(vectorField, vector)
                .field(excludedKeywordField, "label-" + i)
                .endObject()
                .toString();
            for (String index : indices) {
                addKnnDoc(index, docId, doc);
            }
        }
        for (String index : indices) {
            refreshIndex(index);
        }
    }

    private void createSourceExcludeIndex(
        String indexName,
        String vectorField,
        String excludedKeywordField,
        int dimension,
        boolean derivedSourceEnabled,
        boolean withExclude
    ) throws IOException {
        XContentBuilder mapping = XContentFactory.jsonBuilder().startObject();
        if (withExclude) {
            mapping.startObject("_source").array("excludes", excludedKeywordField).endObject();
        }
        mapping.startObject(KNNConstants.PROPERTIES)
            .startObject(vectorField)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .endObject()
            .startObject(excludedKeywordField)
            .field(KNNConstants.TYPE, "keyword")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexName,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", derivedSourceEnabled).build(),
            mapping.toString()
        );
    }

    @SuppressWarnings("unchecked")
    private void assertVectorPresentAndFieldExcluded(String indexName, String vectorField, String excludedField, int dimension, int size)
        throws Exception {
        XContentBuilder searchBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field("size", size)
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject();

        Request searchRequest = new Request("POST", "/" + indexName + "/_search");
        searchRequest.setJsonEntity(searchBuilder.toString());
        Response response = client().performRequest(searchRequest);

        List<Object> hitsList = parseSearchResponseHits(EntityUtils.toString(response.getEntity()));

        assertTrue("Expected at least 10 hits to exercise the sequential fetch path", hitsList.size() >= 10);
        for (Object hitObj : hitsList) {
            Map<String, Object> source = (Map<String, Object>) ((Map<String, Object>) hitObj).get("_source");
            assertTrue(
                String.format(Locale.ROOT, "Vector field '%s' should be reconstructed into _source", vectorField),
                source.containsKey(vectorField)
            );
            assertEquals(
                String.format(Locale.ROOT, "Reconstructed vector '%s' should have full dimension", vectorField),
                dimension,
                ((List<Object>) source.get(vectorField)).size()
            );
            assertFalse(
                String.format(Locale.ROOT, "Excluded field '%s' should be absent from _source", excludedField),
                source.containsKey(excludedField)
            );
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> knnSearchDocIds(String indexName, String vectorField, float[] queryVector, int k) throws Exception {
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(vectorField)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = searchKNNIndex(indexName, queryBuilder, k);
        List<Object> hits = parseSearchResponseHits(EntityUtils.toString(response.getEntity()));

        List<String> docIds = new ArrayList<>();
        for (Object hit : hits) {
            docIds.add((String) ((Map<String, Object>) hit).get("_id"));
        }
        return docIds;
    }

    @SuppressWarnings("unchecked")
    private List<?> extractVector(Map<String, Object> source, String... path) {
        Object current = source;
        for (String key : path) {
            if (current instanceof Map) {
                current = ((Map<String, Object>) current).get(key);
            } else {
                return null;
            }
        }
        if (current instanceof List) {
            return (List<?>) current;
        }
        return null;
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
