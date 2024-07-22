/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.Constants.FIELD_FILTER;
import static org.opensearch.knn.common.Constants.FIELD_TERM;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
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
    private static final String FIELD_NAME_NESTED = "test_nested";
    private static final String FIELD_NAME_VECTOR = "test_vector";
    private static final String FIELD_NAME_PARKING = "parking";
    private static final String FIELD_VALUE_TRUE = "true";
    private static final String FIELD_VALUE_FALSE = "false";
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
    public void testNestedSearchWithLucene_whenKIsTwo_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.LUCENE.getName());

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
    }

    @SneakyThrows
    public void testNestedSearchWithFaiss_whenKIsTwo_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.FAISS.getName());

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
    }

    /**
     * {
     * 	"query": {
     * 		"nested": {
     * 			"path": "test_nested",
     * 			"query": {
     * 				"knn": {
     * 					"test_nested.test_vector": {
     * 						"vector": [
     * 							1, 1, 1
     * 						],
     * 						"k": 3,
     * 						"filter": {
     * 							"term": {
     * 								"parking": "true"
     *                          }
     *                      }
     * 					}
     * 				}
     * 			}
     * 		}
     * 	 }
     * }
     *
     */
    @SneakyThrows
    public void testNestedSearchWithFaiss_whenDoingExactSearch_thenReturnCorrectResults() {
        createKnnIndex(3, KNNEngine.FAISS.getName());

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Make it as an exact search by setting the threshold larger than size of filteredIds(6)
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f };
        Response response = queryNestedField(INDEX_NAME, 3, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, null);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    /**
     * {
     * 	"query": {
     * 		"nested": {
     * 			"path": "test_nested",
     * 			"query": {
     * 				"knn": {
     * 					"test_nested.test_vector": {
     * 						"vector": [
     * 							1, 1, 1
     * 						],
     * 						"min_score": 0.00001,
     * 						"filter": {
     * 							"term": {
     * 								"parking": "true"
     *                          }
     *                      }
     * 					}
     * 				}
     * 			}
     * 		}
     * 	 }
     * }
     *
     */
    @SneakyThrows
    public void testNestedWithFaiss_whenFilter_whenDoRadialSearch_thenReturnCorrectResults() {
        createKnnIndex(3, KNNEngine.FAISS.getName());

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 3f, 3f, 3f };
        Float minScore = 0.00001f;
        Response response = queryNestedField(INDEX_NAME, null, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, minScore);

        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    /**
     * {
     *      "properties": {
     *          "test_nested": {
     *              "type": "nested",
     *              "properties": {
     *                  "test_vector": {
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

    private Response queryNestedField(final String index, final int k, final Object[] vector) throws IOException {
        return queryNestedField(index, k, vector, null, null, null);
    }

    private Response queryNestedField(
        final String index,
        final Integer k,
        final Object[] vector,
        final String filterName,
        final String filterValue,
        final Float minScore
    ) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, FIELD_NAME_NESTED);
        builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
        builder.field(VECTOR, vector);
        if (minScore != null) {
            builder.field(MIN_SCORE, minScore);
        } else if (k != null) {
            builder.field(K, k);
        } else {
            throw new IllegalArgumentException("k or minScore must be provided in the query");
        }
        if (filterName != null && filterValue != null) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);
            builder.field(filterName, filterValue);
            builder.endObject();
            builder.endObject();
        }
        builder.endObject().endObject().endObject().endObject().endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }
}
