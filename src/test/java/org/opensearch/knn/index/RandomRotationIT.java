/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

public class RandomRotationIT extends KNNRestTestCase {

    private static final String TEST_FIELD_NAME = "test-field";

    private String makeQBitIndex(String name, boolean isUnderTest) throws Exception {
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        Integer bits = 1;
        int dimension = 2;
        String indexName = "rand-rot-index" + isUnderTest;
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());

        // Without rotation -> 1,3,2:
        // vec1: --> [1, 0]
        // vec2: --> [0, 1]
        // vec3: --> [0, 0]
        // query: --> [1, 0]

        // With rotation -> 3,1,2
        // vec1: 1, 0 -> [-0.22524017, 0.9743033] --> [0, 1]
        // vec2: 1, 1 -> [0.9743033, 0.22524008] --> [1, 1]
        // vec3: 0, 0 -> [0.22524017, -0.9743033] --> [0, 0]
        // query: 1, 0 -> [-1.0306133, 0.018335745] --> [0, 0]

        // Float[] vector_1 = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // Float[] vector_2 = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        // Float[] vector_3 = { -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        Float[] vector_1 = { 1.0f, 0.0f };
        Float[] vector_2 = { 0.0f, 1.0f };
        Float[] vector_3 = { -1.0f, 0.0f };
        float[] query = { 0.25f, -1.0f };

        addKnnDoc(indexName, "1", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_1));
        addKnnDoc(indexName, "2", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_2));
        addKnnDoc(indexName, "3", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_3));

        forceMergeKnnIndex(indexName);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder();
        queryBuilder.startObject();
        queryBuilder.startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(TEST_FIELD_NAME);
        queryBuilder.field("vector", query);
        queryBuilder.field("k", 3);
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 3).getEntity());
        deleteKNNIndex(indexName);
        return responseBody;
    }

    @SneakyThrows
    public void testRandomRotation() {
        String responseControl = makeQBitIndex(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, false);
        String responseUnderTest = makeQBitIndex(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, true);

        List<Object> controlHits = parseSearchResponseHits(responseControl);
        List<Object> testHits = parseSearchResponseHits(responseUnderTest);

        int controlFirstHitId = Integer.parseInt((String) (((java.util.HashMap<String, Object>) controlHits.get(0)).get("_id")));
        int testFirstHitId = Integer.parseInt((String) (((java.util.HashMap<String, Object>) testHits.get(0)).get("_id")));

        assertEquals(1, controlFirstHitId);
        assertEquals(3, testFirstHitId);
    }

    private void makeOnlyQBitIndex(String indexName, String name, int dimension, int bits, boolean isUnderTest, SpaceType spaceType)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
    }
}
