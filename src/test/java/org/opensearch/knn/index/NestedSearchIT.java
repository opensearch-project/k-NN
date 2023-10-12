/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;

public class NestedSearchIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index-nested-search";
    private static final String FIELD_NAME_NESTED = "test-nested";
    private static final String FIELD_NAME_VECTOR = "test-vector";
    private static final String PROPERTIES_FIELD = "properties";
    private static final int EF_CONSTRUCTION = 128;
    private static final int M = 16;
    private static final SpaceType SPACE_TYPE = SpaceType.L2;

    @After
    @SneakyThrows
    public final void cleanUp() {
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testNestedSearch_whenKIsTwo_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.LUCENE.getName());

        String doc1 = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .add(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 1f, 1f })
            .build();
        addNestedKnnDoc(INDEX_NAME, "1", doc1);

        String doc2 = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .add(FIELD_NAME_VECTOR, new Float[] { 2f, 2f }, new Float[] { 2f, 2f })
            .build();
        addNestedKnnDoc(INDEX_NAME, "2", doc2);

        Float[] queryVector = { 1f, 1f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);

        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            EntityUtils.toString(response.getEntity())
        ).map().get("hits")).get("hits");
        assertEquals(2, hits.size());
    }

    /**
     * {
     *      "properties": {
     *          "test-nested": {
     *              "type": "nested",
     *              "properties": {
     *                  "test-vector": {
     *                      "type": "knn_vector",
     *                      "dimension": 3,
     *                      "method": {
     *                          "name": "hnsw",
     *                          "space_type": "l2",
     *                          "engine": "lucene",
     *                          "parameters": {
     *                              "ef_construction": 128,
     *                              "m": 24
     *                          }
     *                      }
     *                  }
     *              }
     *          }
     *      }
     *  }
     */
    private void createKnnIndex(final int dimension, final String engine) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SPACE_TYPE)
            .field(KNN_ENGINE, engine)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    @SneakyThrows
    private void ingestTestData() {
        String doc1 = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .add(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 1f, 1f })
            .build();
        addNestedKnnDoc(INDEX_NAME, "1", doc1);

        String doc2 = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .add(FIELD_NAME_VECTOR, new Float[] { 2f, 2f }, new Float[] { 2f, 2f })
            .build();
        addNestedKnnDoc(INDEX_NAME, "2", doc2);
    }

    private void addNestedKnnDoc(final String index, final String docId, final String document) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        request.setJsonEntity(document);
        client().performRequest(request);

        request = new Request("POST", "/" + index + "/_refresh");
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private Response queryNestedField(final String index, final int k, final Object[] vector) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, FIELD_NAME_NESTED);
        builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
        builder.field(VECTOR, vector);
        builder.field(K, k);
        builder.endObject().endObject().endObject().endObject().endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    private static class NestedKnnDocBuilder {
        private XContentBuilder builder;

        public NestedKnnDocBuilder(final String fieldName) throws IOException {
            builder = XContentFactory.jsonBuilder().startObject().startArray(fieldName);
        }

        public static NestedKnnDocBuilder create(final String fieldName) throws IOException {
            return new NestedKnnDocBuilder(fieldName);
        }

        public NestedKnnDocBuilder add(final String fieldName, final Object[]... vectors) throws IOException {
            for (Object[] vector : vectors) {
                builder.startObject();
                builder.field(fieldName, vector);
                builder.endObject();
            }
            return this;
        }

        public String build() throws IOException {
            builder.endArray().endObject();
            return builder.toString();
        }

    }
}
