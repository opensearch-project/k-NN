/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ.search;

import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Note that this is simply a sanity test to make sure that concurrent search code path is hit E2E and scores are intact
 * There is no latency verification as it can be better encapsulated in nightly runs.
 */
public class ConcurrentSegmentSearchIT extends KNNRestTestCase {

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (ConcurrentSegmentSearchIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of ConcurrentSegmentSearchIT Class is null");
        }
        URL testIndexVectors = ConcurrentSegmentSearchIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = ConcurrentSegmentSearchIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testConcurrentSegmentSearch_thenSucceed() {
        String indexName = "test-concurrent-segment";
        String fieldName = "test-field-1";
        int dimension = testData.indexData.vectors[0].length;
        final XContentBuilder indexBuilder = createFaissHnswIndexMapping(fieldName, dimension);
        Map<String, Object> mappingMap = xContentBuilderToMap(indexBuilder);
        String mapping = indexBuilder.toString();
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
        refreshAllNonSystemIndices();
        updateIndexSettings(indexName, Settings.builder().put("index.search.concurrent_segment_search.mode", "auto"));

        // Test search queries
        int k = 10;
        verifySearch(indexName, fieldName, k);

        updateIndexSettings(indexName, Settings.builder().put("index.search.concurrent_segment_search.mode", "all"));
        verifySearch(indexName, fieldName, k);
    }

    /*
    {
      "properties": {
        "<fieldName>": {
            "type": "knn_vector",
            "dimension": <dimension>,
            "method": {
            "name": "hnsw",
            "space_type": "l2",
            "engine": "faiss",
            "parameters": {
                "m": 16,
                "ef_construction": 128,
                "ef_search": 128
            }
          }
        }
      }
     */
    @SneakyThrows
    private XContentBuilder createFaissHnswIndexMapping(String fieldName, int dimension) {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .method(
                KNNJsonIndexMappingsBuilder.Method.builder()
                    .engine(KNNEngine.FAISS.getName())
                    .methodName(METHOD_HNSW)
                    .spaceType(SpaceType.L2.getValue())
                    .parameters(KNNJsonIndexMappingsBuilder.Method.Parameters.builder().efConstruction(128).efSearch(128).m(16).build())
                    .build()
            )
            .build()
            .getIndexMappingBuilder();
    }

    @SneakyThrows
    private void verifySearch(String indexName, String fieldName, int k) {
        for (int i = 0; i < testData.queries.length; i++) {
            final KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder().fieldName(fieldName).vector(testData.queries[i]).k(k).build();
            Response response = searchKNNIndex(indexName, queryBuilder, k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), SpaceType.L2),
                    actualScores.get(j),
                    0.0001
                );
            }
        }
    }
}
