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

package org.opensearch.knn.plugin.transport;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

@AllArgsConstructor
public class KNNProfileTransportActionIT extends KNNRestTestCase {
    private String description;
    private SpaceType spaceType;
    private static final String FIELD_NAME = "test-field";
    private static final String INDEX_NAME = "test-index";

    @Before
    public void setup() throws Exception {}

    @ParametersFactory(argumentFormatting = "description:%1$s; spaceType:%2$s")
    public static Collection<Object[]> parameters() {
        return Arrays.asList(
            $$(
                $("SpaceType L2", SpaceType.L2),
                $("SpaceType INNER_PRODUCT", SpaceType.INNER_PRODUCT),
                $("SpaceType COSINESIMIL", SpaceType.COSINESIMIL)
            )
        );
    }

    @SneakyThrows
    public void testProfileEndToEnd() {
        final int dimension = 128;
        final int numDocs = 1000;

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
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        final Settings knnIndexSettings = Settings.builder()
            .put("number_of_shards", 2)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .build();

        createKnnIndex(INDEX_NAME, knnIndexSettings, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        indexTestData(INDEX_NAME, FIELD_NAME, dimension, numDocs);

        Response profileResponse = executeProfileRequest(INDEX_NAME, FIELD_NAME);
        validateProfileResponse(profileResponse, dimension);
    }

    @SneakyThrows
    public void testProfileWithMultipleSegments() {
        final int dimension = 4;
        final int numDocs = 100;

        createKnnIndex(INDEX_NAME, getIndexSettings(), createKnnIndexMapping(FIELD_NAME, dimension));

        for (int i = 0; i < numDocs; i++) {
            float[] vector = new float[dimension];
            Arrays.fill(vector, (float) i);
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);

            if (i % 10 == 0) {
                refreshIndex(INDEX_NAME);
            }
        }

        refreshIndex(INDEX_NAME);

        Response profileResponse = executeProfileRequest(INDEX_NAME, FIELD_NAME);
        Map<String, Object> responseMap = parseResponse(profileResponse);

        Map<String, Object> shardProfiles = (Map<String, Object>) responseMap.get("shard_profiles");
        for (Object shardProfile : shardProfiles.values()) {
            List<Map<String, Object>> segments = (List<Map<String, Object>>) ((Map<String, Object>) shardProfile).get("segments");
            assertTrue("Should have multiple segments", segments.size() > 1);
        }
    }

    private void indexTestData(String indexName, String fieldName, int dimension, int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] vector = new float[dimension];
            Arrays.fill(vector, (float) i);
            addKnnDoc(indexName, Integer.toString(i), fieldName, vector);
        }

        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }

    @SneakyThrows
    protected Response executeProfileRequest(String indexName, String fieldName) throws IOException {
        Request request = new Request("GET", "/_plugins/_knn/profile/" + indexName + "/" + fieldName);
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return response;
    }

    private void validateProfileResponse(Response response, int dimension) throws IOException, ParseException {
        Map<String, Object> responseMap = parseResponse(response);

        assertNotNull(responseMap.get("total_shards"));
        assertNotNull(responseMap.get("successful_shards"));
        assertNotNull(responseMap.get("failed_shards"));

        Map<String, Object> shardProfiles = (Map<String, Object>) responseMap.get("shard_profiles");
        assertFalse(shardProfiles.isEmpty());

        Map<String, Object> clusterAgg = (Map<String, Object>) responseMap.get("cluster_aggregation");
        assertEquals(dimension, clusterAgg.get("dimension"));

        List<Map<String, Object>> dimensions = (List<Map<String, Object>>) clusterAgg.get("dimensions");
        assertEquals(dimension, dimensions.size());

        for (Map<String, Object> dimStats : dimensions) {
            validateDimensionStatistics(dimStats);
        }
    }

    private void validateDimensionStatistics(Map<String, Object> dimStats) {
        assertNotNull("Count should be present", dimStats.get("count"));
        assertNotNull("Min should be present", dimStats.get("min"));
        assertNotNull("Max should be present", dimStats.get("max"));
        assertNotNull("Mean should be present", dimStats.get("mean"));
        assertNotNull("Standard deviation should be present", dimStats.get("std_deviation"));
        assertNotNull("Variance should be present", dimStats.get("variance"));
    }

    private Map<String, Object> parseResponse(Response response) throws IOException, ParseException {
        return createParser(XContentType.JSON.xContent(), EntityUtils.toString(response.getEntity())).map();
    }

    private Settings getIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();
    }
}
