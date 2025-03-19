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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.*;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD;

public class RemoteBuildIT extends KNNRestTestCase {
    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (RemoteBuildIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of RemoteBuildIT Class is null");
        }
        URL testIndexVectors = RemoteBuildIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = RemoteBuildIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    @Before
    public void setupAdditionalRemoteIndexBuildSettings() throws Exception {
        final String remoteBuild = System.getProperty("test.remoteBuild", null);
        if (isRemoteIndexBuildSupported(getBWCVersion()) && remoteBuild != null) {
            updateClusterSettings(KNNFeatureFlags.KNN_REMOTE_VECTOR_BUILD_SETTING.getKey(), true);
            updateClusterSettings(KNNSettings.KNN_REMOTE_VECTOR_REPO, "integ-test-repo");
            updateClusterSettings(KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT, "http://0.0.0.0:80");
            updateClusterSettings(KNNSettings.KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL, 0);
            setupRepository("integ-test-repo");
        }
    }

    @SneakyThrows
    public void testEndToEnd_whenDoRadiusSearch_whenDistanceThreshold_whenMethodIsHNSWFlat_thenSucceed() {
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        final Settings knnIndexSettings = buildKNNIndexSettingsRemoteBuild(0);
        createKnnIndex(INDEX_NAME, knnIndexSettings, mapping);

        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        float distance = 300000000000f;
        validateRadiusSearchResults(INDEX_NAME, FIELD_NAME, testData.queries, distance, null, spaceType, null, null);

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenGraphThresholdIsMetDuringMerge_thenCreateGraph() {
        final String indexName = "test-index-hnsw-sqfp16";
        final String fieldName = "test-field-hnsw-sqfp16";
        final SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
        final Random random = new Random();
        final SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];
        final int dimension = 128;
        final int numDocs = 100;

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        final Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        final String mapping = builder.toString();

        final Settings knnIndexSettings = buildKNNIndexSettingsRemoteBuild(numDocs);
        createKnnIndex(indexName, knnIndexSettings, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));
        indexTestData(indexName, fieldName, dimension, numDocs);

        final float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        // Assert we have the right number of documents in the index
        assertEquals(numDocs, getDocCount(indexName));

        // KNN Query should return empty result
        final Response searchResponse = searchKNNIndex(indexName, buildSearchQuery(fieldName, 1, queryVector, null), 1);
        final List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(1, results.size());

        // update index setting to build graph and do force merge
        // update build vector data structure setting
        forceMergeKnnIndex(indexName, 1);

        queryTestData(indexName, fieldName, dimension, numDocs);

        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    private void validateGraphEviction() throws Exception {
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

    private void indexTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] indexVector = new float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDocWithAttributes(indexName, Integer.toString(i), fieldName, indexVector, ImmutableMap.of("rating", String.valueOf(i)));
        }

        // Assert that all docs are ingested
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }

    @SneakyThrows
    private void queryTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) {
        queryTestData(indexName, fieldName, dimension, numDocs, null);
    }

    private void queryTestData(
        final String indexName,
        final String fieldName,
        final int dimension,
        final int numDocs,
        Map<String, ?> methodParams
    ) throws IOException, ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);
        int k = 10;

        Response searchResponse = searchKNNIndex(indexName, buildSearchQuery(fieldName, k, queryVector, methodParams), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    private List<List<KNNResult>> validateRadiusSearchResults(
        String indexName,
        String fieldName,
        float[][] queryVectors,
        Float distanceThreshold,
        Float scoreThreshold,
        final SpaceType spaceType,
        TermQueryBuilder filterQuery,
        Map<String, ?> methodParameters
    ) throws IOException, ParseException {
        List<List<KNNResult>> queryResults = new ArrayList<>();
        for (float[] queryVector : queryVectors) {
            XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
            queryBuilder.startObject("knn");
            queryBuilder.startObject(fieldName);
            queryBuilder.field("vector", queryVector);
            if (distanceThreshold != null) {
                queryBuilder.field(MAX_DISTANCE, distanceThreshold);
            } else if (scoreThreshold != null) {
                queryBuilder.field(MIN_SCORE, scoreThreshold);
            } else {
                throw new IllegalArgumentException("Invalid threshold");
            }
            if (filterQuery != null) {
                queryBuilder.field("filter", filterQuery);
            }
            if (methodParameters != null && methodParameters.size() > 0) {
                queryBuilder.startObject(METHOD_PARAMETER);
                for (Map.Entry<String, ?> entry : methodParameters.entrySet()) {
                    queryBuilder.field(entry.getKey(), entry.getValue());
                }
                queryBuilder.endObject();
            }
            queryBuilder.endObject();
            queryBuilder.endObject();
            queryBuilder.endObject().endObject();
            final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 10).getEntity());

            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);

            for (KNNResult knnResult : knnResults) {
                float[] vector = knnResult.getVector();
                float distance = TestUtils.computeDistFromSpaceType(spaceType, vector, queryVector);
                if (spaceType == SpaceType.L2) {
                    assertTrue(KNNScoringUtil.l2Squared(queryVector, vector) <= distance);
                } else if (spaceType == SpaceType.INNER_PRODUCT) {
                    assertTrue(KNNScoringUtil.innerProduct(queryVector, vector) >= distance);
                } else if (spaceType == SpaceType.COSINESIMIL) {
                    assertTrue(KNNScoringUtil.cosinesimil(queryVector, vector) >= distance);
                } else {
                    throw new IllegalArgumentException("Invalid space type");
                }
            }
            queryResults.add(knnResults);
        }
        return queryResults;
    }

    @SneakyThrows
    protected void setupRepository(String repository) {
        final String bucket = System.getProperty("test.bucket", null);
        final String base_path = System.getProperty("test.base_path", null);

        Settings.Builder builder = Settings.builder()
            .put("bucket", bucket)
            .put("base_path", base_path)
            .put("region", "us-east-1")
            .put("s3_upload_retry_enabled", false);

        final String remoteBuild = System.getProperty("test.remoteBuild", null);
        if (remoteBuild != null && remoteBuild.equals("s3.localStack")) {
            builder.put("endpoint", "http://s3.localhost.localstack.cloud:4566");
        }

        registerRepository(repository, "s3", false, builder.build());

    }

    protected Settings buildKNNIndexSettingsRemoteBuild(int approximateThreshold) {
        Settings.Builder builder = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, approximateThreshold);

        final String remoteBuild = System.getProperty("test.remoteBuild", null);

        if (isRemoteIndexBuildSupported(getBWCVersion()) && remoteBuild != null) {
            builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD, true);
            builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "1kb");
        }
        return builder.build();
    }
}
