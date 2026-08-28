/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.google.common.collect.ImmutableList;
import com.google.common.primitives.Floats;
import org.opensearch.common.settings.Settings;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

@AllArgsConstructor
public class FaissHNSWFlatE2EIT extends KNNRestTestCase {

    private String description;
    private int k;
    private Map<String, ?> methodParameters;
    private boolean deleteRandomDocs;

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (FaissHNSWFlatE2EIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of FaissIT Class is null");
        }
        URL testIndexVectors = FaissHNSWFlatE2EIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = FaissHNSWFlatE2EIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    @ParametersFactory(argumentFormatting = "description:%1$s; k:%2$s; efSearch:%3$s, deleteDocs:%4$s")
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            $$(
                $("Valid k, valid efSearch efSearch value", 10, Map.of(METHOD_PARAMETER_EF_SEARCH, 300), false),
                $("Valid k, efsearch absent", 10, null, false),
                $("Has delete docs, ef_search", 10, Map.of(METHOD_PARAMETER_EF_SEARCH, 300), true),
                $("Has delete docs", 10, null, true)
            )
        );
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testEndToEnd_whenMethodIsHNSWFlat_thenSucceed() {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents in the index
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        // Delete few Docs
        if (deleteRandomDocs) {
            final Set<Integer> docIdsToBeDeleted = new HashSet<>();
            while (docIdsToBeDeleted.size() < 10) {
                docIdsToBeDeleted.add(randomInt(testData.indexData.docs.length - 1));
            }

            for (Integer id : docIdsToBeDeleted) {
                deleteKnnDoc(indexName, Integer.toString(testData.indexData.docs[id]));
            }
            refreshAllNonSystemIndices();
            forceMergeKnnIndex(indexName, 3);

            assertEquals(testData.indexData.docs.length - 10, getDocCount(indexName));
        }

        // Test search queries
        for (int i = 0; i < testData.queries.length; i++) {
            final KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(testData.queries[i])
                .k(k)
                .methodParameters(methodParameters)
                .build();
            Response response = searchKNNIndex(indexName, queryBuilder, k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }
            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    // Exercises FP32 flat-vector dedup on the DEFAULT (non-MOS) search path: a graph-only .faiss (no embedded flat
    // storage) is written, then bulk-loaded into native memory where the flat storage is reconstructed by streaming
    // full-precision vectors from Lucene's .vec file. Scores must remain exact (identical to a full build), which
    // verifies the reconstructed storage and its ordinal ordering against the graph.
    @SneakyThrows
    public void testEndToEnd_whenFlatVectorDedup_thenSucceed() {
        String indexName = "test-index-flat-vector-dedup";
        String fieldName = "test-field-1";
        SpaceType spaceType = SpaceType.L2;
        Integer dimension = testData.indexData.vectors[0].length;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, 16)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();

        // Enable flat-vector dedup (graph-only .faiss). Memory-optimized search is left disabled, so search bulk-loads
        // the index into native memory and reconstructs the flat storage from .vec - the non-MOS path under test.
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_FLAT_VECTOR_DEDUP, true)
            .build();
        createKnnIndex(indexName, settings, mapping);

        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        if (deleteRandomDocs) {
            final Set<Integer> docIdsToBeDeleted = new HashSet<>();
            while (docIdsToBeDeleted.size() < 10) {
                docIdsToBeDeleted.add(randomInt(testData.indexData.docs.length - 1));
            }
            for (Integer id : docIdsToBeDeleted) {
                deleteKnnDoc(indexName, Integer.toString(testData.indexData.docs[id]));
            }
            refreshAllNonSystemIndices();
            // Force-merge to consolidate and purge deletes, producing a merged graph-only .faiss whose native
            // reconstruction from .vec is exercised on search. Mirrors the non-deduped end-to-end test.
            forceMergeKnnIndex(indexName, 3);
            assertEquals(testData.indexData.docs.length - 10, getDocCount(indexName));
        }

        // Even without a force-merge, flushed segments are written graph-only (dedup) and are bulk-loaded into native
        // memory on search, exercising the reconstruction path.
        for (int i = 0; i < testData.queries.length; i++) {
            final KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(testData.queries[i])
                .k(k)
                .methodParameters(methodParameters)
                .build();
            Response response = searchKNNIndex(indexName, queryBuilder, k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);

            // Exact-score check first: catches any ordinal misalignment in the reconstructed storage. Each returned
            // result's search score must equal the exact L2 score of its own (source) vector.
            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < knnResults.size(); j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
            assertEquals(k, knnResults.size());
        }

        deleteKNNIndex(indexName);
    }
}
