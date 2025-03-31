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
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
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
import java.util.TreeMap;
import java.util.Collection;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD;

@AllArgsConstructor
public class RemoteBuildIT extends KNNRestTestCase {
    private static TestUtils.TestData testData;
    private String description;
    private SpaceType spaceType;

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
        updateClusterSettings(KNNFeatureFlags.KNN_REMOTE_VECTOR_BUILD_SETTING.getKey(), true);
        updateClusterSettings(KNNSettings.KNN_REMOTE_VECTOR_REPO, "integ-test-repo");
        updateClusterSettings(KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT, "http://0.0.0.0:80");
        updateClusterSettings(KNNSettings.KNN_REMOTE_BUILD_CLIENT_POLL_INTERVAL, 0);
        setupRepository("integ-test-repo");
    }

    @ParametersFactory(argumentFormatting = "description:%1$s; spaceType:%2$s")
    public static Collection<Object[]> parameters() throws IOException {
        return Arrays.asList(
            $$(
                $("SpaceType L2", SpaceType.L2),
                $("SpaceType INNER_PRODUCT", SpaceType.INNER_PRODUCT),
                $("SpaceType COSINESIMIL", SpaceType.COSINESIMIL)
            )
        );
    }

    @SneakyThrows
    public void testEndToEnd_whenDoRadiusSearch_whenDistanceThreshold_whenMethodIsHNSWFlat_thenSucceed() {
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

        forceMergeKnnIndex(INDEX_NAME, 1);
        validateRadiusSearchResults(INDEX_NAME, FIELD_NAME, testData.queries, distance, null, spaceType, null, null);

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testFilteredSearchWithFaissHnsw_whenFiltersMatchAllDocs_thenReturnCorrectResults() {
        String filterFieldName = "color";
        final int expectResultSize = randomIntBetween(1, 3);
        final String filterValue = "red";
        final Settings knnIndexSettings = buildKNNIndexSettingsRemoteBuild(0);
        createKnnIndex(INDEX_NAME, knnIndexSettings, createKnnIndexMapping(FIELD_NAME, 3, METHOD_HNSW, FAISS_NAME, spaceType.getValue()));

        // ingest 5 vector docs into the index with the same field {"color": "red"}
        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i + 1, i + 1, i + 1 }, ImmutableMap.of(filterFieldName, filterValue));
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 0));

        Float[] queryVector = { 3f, 3f, 3f };
        // All docs in one segment will match the filters value
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(expectResultSize)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(INDEX_NAME, query, expectResultSize);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(expectResultSize, docIds.size());
        assertEquals(expectResultSize, parseTotalSearchHits(entity));
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
        return Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, approximateThreshold)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD, true)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "0kb")
            .build();
    }
}
