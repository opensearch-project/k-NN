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
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.hamcrest.Matchers.containsString;

public class NmslibIT extends KNNRestTestCase {

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (NmslibIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of NmslibIT Class is null");
        }
        URL testIndexVectors = NmslibIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = NmslibIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    public void testInvalidMethodParameters() throws Exception {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";
        Integer dimension = testData.indexData.vectors[0].length;
        KNNMethod hnswMethod = KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L1;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, 32)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 100)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        final Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        // Index the test data
        // Adding only doc to cut on integ test time
        addKnnDoc(
            indexName,
            Integer.toString(testData.indexData.docs[0]),
            fieldName,
            Floats.asList(testData.indexData.vectors[0]).toArray()
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> searchKNNIndex(
                indexName,
                KNNQueryBuilder.builder()
                    .k(10)
                    .methodParameters(Map.of("foo", "bar"))
                    .vector(testData.queries[0])
                    .fieldName(fieldName)
                    .build(),
                10
            )
        );
        expectThrows(
            IllegalArgumentException.class,
            () -> searchKNNIndex(
                indexName,
                KNNQueryBuilder.builder()
                    .k(10)
                    .methodParameters(Map.of("ef_search", "bar"))
                    .vector(testData.queries[0])
                    .fieldName(fieldName)
                    .build(),
                10
            )
        );
    }

    public void testEndToEnd() throws Exception {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        KNNMethod hnswMethod = KNNEngine.NMSLIB.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L1;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.NMSLIB.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
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
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        // search index
        // without method parameters
        validateSearch(indexName, fieldName, spaceType, null);
        // With valid method params
        validateSearch(indexName, fieldName, spaceType, Map.of("ef_search", 50));

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

    @SneakyThrows
    private void validateSearch(
        final String indexName,
        final String fieldName,
        SpaceType spaceType,
        final Map<String, Object> methodParams
    ) {
        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            Response response = searchKNNIndex(
                indexName,
                KNNQueryBuilder.builder().fieldName(fieldName).vector(testData.queries[i]).k(k).methodParameters(methodParams).build(),
                k
            );
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.NMSLIB.score(KNNScoringUtil.l1Norm(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }
    }

    public void testAddDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
    }

    public void testUpdateDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // update
        Float[] updatedVector = { 8.0f, 8.0f };
        updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, updatedVector);
    }

    public void testDeleteDoc() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // delete knn doc
        deleteKnnDoc(INDEX_NAME, "1");
    }

    public void testCreateIndexWithValidAlgoParams_settings() {
        try {
            Settings settings = Settings.builder()
                .put(getKNNDefaultIndexSettings())
                .put("index.knn.algo_param.m", 32)
                .put("index.knn.algo_param.ef_construction", 400)
                .build();
            createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));
            Float[] vector = { 6.0f, 6.0f };
            addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
        } catch (Exception ex) {
            fail("Exception not expected as valid index arguements passed: " + ex);
        }
    }

    @SuppressWarnings("unchecked")
    public void testCreateIndexWithValidAlgoParams_mapping() {
        try {
            Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

            String spaceType = SpaceType.L1.getValue();
            int efConstruction = 14;
            int m = 13;

            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", 2)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
                .field(KNNConstants.METHOD_PARAMETER_M, m)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(INDEX_NAME, settings, mapping);

            Map<String, Object> fullMapping = getAsMap(INDEX_NAME + "/_mapping");
            Map<String, Object> indexMapping = (Map<String, Object>) fullMapping.get(INDEX_NAME);
            Map<String, Object> mappingsMapping = (Map<String, Object>) indexMapping.get("mappings");
            Map<String, Object> propertiesMapping = (Map<String, Object>) mappingsMapping.get("properties");
            Map<String, Object> fieldMapping = (Map<String, Object>) propertiesMapping.get(FIELD_NAME);
            Map<String, Object> methodMapping = (Map<String, Object>) fieldMapping.get(KNNConstants.KNN_METHOD);
            Map<String, Object> parametersMapping = (Map<String, Object>) methodMapping.get(KNNConstants.PARAMETERS);
            String spaceTypeMapping = (String) methodMapping.get(KNNConstants.METHOD_PARAMETER_SPACE_TYPE);
            Integer mMapping = (Integer) parametersMapping.get(KNNConstants.METHOD_PARAMETER_M);
            Integer efConstructionMapping = (Integer) parametersMapping.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);

            assertEquals(spaceType, spaceTypeMapping);
            assertEquals(m, mMapping.intValue());
            assertEquals(efConstruction, efConstructionMapping.intValue());

            Float[] vector = { 6.0f, 6.0f };
            addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
        } catch (Exception ex) {
            fail("Exception not expected as valid index arguments passed: " + ex);
        }
    }

    public void testCreateIndexWithValidAlgoParams_mappingAndSettings() {
        try {
            String spaceType1 = SpaceType.L1.getValue();
            int efConstruction1 = 14;
            int m1 = 13;

            Settings settings = Settings.builder()
                .put(getKNNDefaultIndexSettings())
                .put("index.knn.algo_param.m", m1)
                .put("index.knn.algo_param.ef_construction", efConstruction1)
                .build();

            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", 2)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType1)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction1)
                .field(KNNConstants.METHOD_PARAMETER_M, m1)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(INDEX_NAME + "1", settings, mapping);
            Float[] vector = { 6.0f, 6.0f };
            addKnnDoc(INDEX_NAME + "1", "1", FIELD_NAME, vector);

            String spaceType2 = SpaceType.COSINESIMIL.getValue();
            int efConstruction2 = 114;
            int m2 = 113;

            mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME + "1")
                .field("type", "knn_vector")
                .field("dimension", 2)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType1)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction1)
                .field(KNNConstants.METHOD_PARAMETER_M, m1)
                .endObject()
                .endObject()
                .endObject()
                .startObject(FIELD_NAME + "2")
                .field("type", "knn_vector")
                .field("dimension", 2)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType2)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction2)
                .field(KNNConstants.METHOD_PARAMETER_M, m2)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(INDEX_NAME + "2", settings, mapping);
            addKnnDoc(INDEX_NAME + "2", "1", FIELD_NAME, vector);
        } catch (Exception ex) {
            fail("Exception not expected as valid index arguments passed: " + ex);
        }
    }

    public void testQueryIndexWithValidQueryAlgoParams() throws IOException {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).put("index.knn.algo_param.ef_search", 300).build();
        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        float[] queryVector = { 1.0f, 1.0f }; // vector to be queried
        int k = 1; // nearest 1 neighbor
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, k);
        searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
    }

    public void testInvalidIndexHnswAlgoParams_settings() {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).put("index.knn.algo_param.m", "-1").build();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2)));
    }

    public void testInvalidIndexHnswAlgoParams_mapping() throws IOException {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 2)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, "-1")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, mapping));
    }

    public void testInvalidIndexHnswAlgoParams_mappingAndSettings() throws IOException {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).put("index.knn.algo_param.m", "-1").build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 2)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, "-1")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, mapping));
    }

    public void testInvalidQueryHnswAlgoParams() {
        Settings settings = Settings.builder().put(getKNNDefaultIndexSettings()).put("index.knn.algo_param.ef_search", "-1").build();
        Exception ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2))
        );
        assertThat(ex.getMessage(), containsString("Failed to parse value [-1] for setting [index.knn.algo_param.ef_search]"));
    }

    @Override
    protected Settings restClientSettings() {
        return noStrictDeprecationModeSettingsBuilder().build();
    }
}
