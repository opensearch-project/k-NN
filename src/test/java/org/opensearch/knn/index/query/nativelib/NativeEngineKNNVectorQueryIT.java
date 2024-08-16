/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.google.common.primitives.Floats;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.apache.http.util.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.FaissHNSWFlatE2EIT;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

@AllArgsConstructor
public class NativeEngineKNNVectorQueryIT extends KNNRestTestCase {

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
                $("test without deletedocs", 10, Map.of(METHOD_PARAMETER_EF_SEARCH, 300), false),
                $("test with deletedocs", 10, Map.of(METHOD_PARAMETER_EF_SEARCH, 300), true)
            )
        );
    }

    @SneakyThrows
    public void testResultComparisonSanity() {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        SpaceType spaceType = SpaceType.L2;

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
            .field(KNNConstants.METHOD_PARAMETER_M, 16)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 32)
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, 32)
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

        int queryIndex = ThreadLocalRandom.current().nextInt(testData.queries.length);
        // Test search queries
        final KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(testData.queries[queryIndex])
            .k(k)
            .methodParameters(methodParameters)
            .build();
        Response nativeEngineResponse = searchKNNIndex(indexName, queryBuilder, k);
        String responseBody = EntityUtils.toString(nativeEngineResponse.getEntity());
        List<KNNResult> nativeEngineKnnResults = parseSearchResponse(responseBody, fieldName);
        assertEquals(k, nativeEngineKnnResults.size());

        List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
        for (int j = 0; j < k; j++) {
            float[] primitiveArray = nativeEngineKnnResults.get(j).getVector();
            assertEquals(
                KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[queryIndex], primitiveArray), spaceType),
                actualScores.get(j),
                0.0001
            );
        }

        updateClusterSettings("knn.feature.query.rewrite.enabled", false);
        Response launchControlDisabledResponse = searchKNNIndex(indexName, queryBuilder, k);
        String launchControlDisabledResponseString = EntityUtils.toString(launchControlDisabledResponse.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(launchControlDisabledResponseString, fieldName);
        assertEquals(k, knnResults.size());

        assertEquals(nativeEngineKnnResults, knnResults);

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
}
