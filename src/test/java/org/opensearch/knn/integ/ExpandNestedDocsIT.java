/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.google.common.collect.Multimap;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
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
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static com.carrotsearch.randomizedtesting.RandomizedTest.$$;
import static org.opensearch.knn.common.Constants.FIELD_FILTER;
import static org.opensearch.knn.common.Constants.FIELD_TERM;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

@Log4j2
@AllArgsConstructor
public class ExpandNestedDocsIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index-expand-nested-search";
    private static final String FIELD_NAME_NESTED = "test_nested";
    private static final String FIELD_NAME_VECTOR = "test_vector";
    private static final String FIELD_NAME_PARKING = "parking";
    private static final String FIELD_NAME_STORAGE = "storage";
    private static final String TYPE_BOOLEAN = "boolean";
    private static final String FIELD_VALUE_TRUE = "true";
    private static final String FIELD_VALUE_FALSE = "false";
    private static final String PROPERTIES_FIELD = "properties";
    private static final String INNER_HITS = "inner_hits";

    private String description;
    private KNNEngine engine;
    private VectorDataType dataType;
    private Mode mode;
    private Integer dimension;

    @After
    @SneakyThrows
    public final void cleanUp() {
        deleteKNNIndex(INDEX_NAME);
    }

    @ParametersFactory(argumentFormatting = "description:%1$s; engine:%2$s, data_type:%3$s, mode:%4$s, dimension:%5$s")
    public static Collection<Object[]> parameters() throws IOException {
        int dimension = 1;
        return Arrays.asList(
            $$(
                $("Lucene with byte format and in memory mode", KNNEngine.LUCENE, VectorDataType.BYTE, Mode.NOT_CONFIGURED, dimension),
                $("Lucene with float format and in memory mode", KNNEngine.LUCENE, VectorDataType.FLOAT, Mode.NOT_CONFIGURED, dimension),
                $(
                    "Faiss with binary format and in memory mode",
                    KNNEngine.FAISS,
                    VectorDataType.BINARY,
                    Mode.NOT_CONFIGURED,
                    dimension * 8
                ),
                $("Faiss with byte format and in memory mode", KNNEngine.FAISS, VectorDataType.BYTE, Mode.NOT_CONFIGURED, dimension),
                $("Faiss with float format and in memory mode", KNNEngine.FAISS, VectorDataType.FLOAT, Mode.IN_MEMORY, dimension),
                $(
                    "Faiss with float format and on disk mode",
                    KNNEngine.FAISS,
                    VectorDataType.FLOAT,
                    Mode.ON_DISK,
                    // Currently, on disk mode only supports dimension of multiple of 8
                    dimension * 8
                )
            )
        );
    }

    @SneakyThrows
    public void testExpandNestedDocs_whenFilteredOnParentDoc_thenReturnAllNestedDoc() {
        int numberOfNestedFields = 2;
        createKnnIndex(engine, mode, dimension, dataType);
        addRandomVectorsWithTopLevelField(1, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);
        addRandomVectorsWithTopLevelField(2, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);
        addRandomVectorsWithTopLevelField(3, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);
        addRandomVectorsWithTopLevelField(4, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);
        addRandomVectorsWithTopLevelField(5, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);
        deleteKnnDoc(INDEX_NAME, String.valueOf(1));
        updateVectorWithTopLevelField(2, numberOfNestedFields, FIELD_NAME_PARKING, FIELD_VALUE_FALSE);

        // Run
        Float[] queryVector = createVector();
        Response response = queryNestedFieldWithExpandNestedDocs(INDEX_NAME, 10, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE);

        // Verify
        String entity = EntityUtils.toString(response.getEntity());
        Multimap<String, Integer> docIdToOffsets = parseInnerHits(entity, FIELD_NAME_NESTED);
        assertEquals(3, docIdToOffsets.keySet().size());
        for (String key : docIdToOffsets.keySet()) {
            assertEquals(numberOfNestedFields, docIdToOffsets.get(key).size());
        }
    }

    @SneakyThrows
    public void testExpandNestedDocs_whenFilteredOnNestedFieldDoc_thenReturnFilteredNestedDoc() {
        int numberOfNestedFields = 2;
        createKnnIndex(engine, mode, dimension, dataType);
        addRandomVectorsWithMetadata(1, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_FALSE, FIELD_VALUE_FALSE));
        addRandomVectorsWithMetadata(2, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_TRUE, FIELD_VALUE_TRUE));
        addRandomVectorsWithMetadata(3, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_TRUE, FIELD_VALUE_TRUE));
        addRandomVectorsWithMetadata(4, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_FALSE, FIELD_VALUE_TRUE));
        addRandomVectorsWithMetadata(5, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_TRUE, FIELD_VALUE_FALSE));
        deleteKnnDoc(INDEX_NAME, String.valueOf(1));
        addRandomVectorsWithMetadata(2, numberOfNestedFields, FIELD_NAME_STORAGE, Arrays.asList(FIELD_VALUE_FALSE, FIELD_VALUE_FALSE));

        // Run
        Float[] queryVector = createVector();
        Response response = queryNestedFieldWithExpandNestedDocs(
            INDEX_NAME,
            10,
            queryVector,
            FIELD_NAME_NESTED + "." + FIELD_NAME_STORAGE,
            FIELD_VALUE_TRUE
        );

        // Verify
        String entity = EntityUtils.toString(response.getEntity());
        Multimap<String, Integer> docIdToOffsets = parseInnerHits(entity, FIELD_NAME_NESTED);
        assertEquals(3, docIdToOffsets.keySet().size());
        assertEquals(2, docIdToOffsets.get(String.valueOf(3)).size());
        assertEquals(1, docIdToOffsets.get(String.valueOf(4)).size());
        assertEquals(1, docIdToOffsets.get(String.valueOf(5)).size());

        assertTrue(docIdToOffsets.get(String.valueOf(4)).contains(1));
        assertTrue(docIdToOffsets.get(String.valueOf(5)).contains(0));
    }

    @SneakyThrows
    public void testExpandNestedDocs_whenMultiShards_thenReturnCorrectResult() {
        int numberOfNestedFields = 10;
        int numberOfDocuments = 5;
        createKnnIndex(engine, mode, dimension, dataType, 2);
        for (int i = 1; i <= numberOfDocuments; i++) {
            addSingleRandomVectors(i, numberOfNestedFields);
        }
        forceMergeKnnIndex(INDEX_NAME);

        // Run
        Float[] queryVector = createVector();
        Response response = queryNestedFieldWithExpandNestedDocs(INDEX_NAME, numberOfDocuments, queryVector);

        // Verify
        String entity = EntityUtils.toString(response.getEntity());
        Multimap<String, Integer> docIdToOffsets = parseInnerHits(entity, FIELD_NAME_NESTED);
        assertEquals(numberOfDocuments, docIdToOffsets.keySet().size());
        int defaultInnerHitSize = 3;
        for (int i = 1; i <= numberOfDocuments; i++) {
            assertEquals(defaultInnerHitSize, docIdToOffsets.get(String.valueOf(i)).size());
        }
    }

    private Float[] createVector() {
        int vectorSize = VectorDataType.BINARY.equals(dataType) ? dimension / 8 : dimension;
        Float[] vector = new Float[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            vector[i] = (float) (randomInt(255) - 128);
        }
        return vector;
    }

    private void updateVectorWithTopLevelField(
        final int docId,
        final int numOfNestedFields,
        final String fieldName,
        final String fieldValue
    ) throws IOException {
        addRandomVectorsWithTopLevelField(docId, numOfNestedFields, fieldName, fieldValue);
    }

    private void addRandomVectorsWithTopLevelField(
        final int docId,
        final int numOfNestedFields,
        final String fieldName,
        final String fieldValue
    ) throws IOException {

        NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(FIELD_NAME_NESTED);
        for (int i = 0; i < numOfNestedFields; i++) {
            builder.addVectors(FIELD_NAME_VECTOR, createVector());
        }
        builder.addTopLevelField(fieldName, fieldValue);
        String doc = builder.build();
        addKnnDoc(INDEX_NAME, String.valueOf(docId), doc);
        refreshIndex(INDEX_NAME);
    }

    private void addSingleRandomVectors(final int docId, final int numOfNestedFields) throws IOException {
        NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(FIELD_NAME_NESTED);
        Object[] vector = createVector();
        for (int i = 0; i < numOfNestedFields; i++) {
            builder.addVectors(FIELD_NAME_VECTOR, vector);
        }
        String doc = builder.build();
        addKnnDoc(INDEX_NAME, String.valueOf(docId), doc);
        refreshIndex(INDEX_NAME);
    }

    private void addRandomVectorsWithMetadata(
        final int docId,
        final int numOfNestedFields,
        final String nestedFieldName,
        final List<String> nestedFieldValue
    ) throws IOException {
        assert numOfNestedFields == nestedFieldValue.size();

        NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(FIELD_NAME_NESTED);
        for (int i = 0; i < numOfNestedFields; i++) {
            builder.addVectorWithMetadata(FIELD_NAME_VECTOR, createVector(), nestedFieldName, nestedFieldValue.get(i));
        }
        String doc = builder.build();
        addKnnDoc(INDEX_NAME, String.valueOf(docId), doc);
        refreshIndex(INDEX_NAME);
    }

    private void createKnnIndex(final KNNEngine engine, final Mode mode, final int dimension, final VectorDataType vectorDataType)
        throws Exception {
        createKnnIndex(engine, mode, dimension, vectorDataType, 1);
    }

    /**
     * {
     * 		"dynamic": false,
     *      "properties": {
     *          "test_nested": {
     *              "type": "nested",
     *              "properties": {
     *                  "test_vector": {
     *                      "type": "knn_vector",
     *                      "dimension": 3,
     *                      "mode": "in_memory",
     *                      "data_type: "float",
     *                      "method": {
     *                          "name": "hnsw",
     *                          "engine": "lucene"
     *                      }
     *                  },
     *                  "storage": {
     *                      "type": "boolean"
     *                  }
     *              }
     *          },
     *          "parking": {
     *              "type": "boolean"
     *          }
     *      }
     *  }
     */
    private void createKnnIndex(
        final KNNEngine engine,
        final Mode mode,
        final int dimension,
        final VectorDataType vectorDataType,
        final int numOfShards
    ) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field("dynamic", false)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .field(MODE_PARAMETER, Mode.NOT_CONFIGURED.equals(mode) ? null : mode.getName())
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine.getName())
            .endObject()
            .endObject()
            .startObject(FIELD_NAME_STORAGE)
            .field(TYPE, TYPE_BOOLEAN)
            .endObject()
            .endObject()
            .endObject()
            .startObject(FIELD_NAME_PARKING)
            .field(TYPE, TYPE_BOOLEAN)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        Settings settings = Settings.builder()
            .put("number_of_shards", numOfShards)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private Response queryNestedFieldWithExpandNestedDocs(final String index, final Integer k, final Object[] vector) throws IOException {
        return queryNestedFieldWithExpandNestedDocs(index, k, vector, null, null);
    }

    /**
     * {
     *      "query": {
     *          "nested": {
     *              "path": "test_nested",
     *              "query": {
     *                  "knn": {
     *                      "test_nested.test_vector" : {
     *                          "vector: [1, 1, 2]
     *                       	"k": 3,
     *                      	"filter": {
     *                      	 	"term": {
     *                      	 		"nested_field.storage": true
     *                      	 	}
     *                      	}
     *                      }
    *                      }
     *          	},
     *          	"inner_hits": {}
     *          }
     *      }
     *  }
     */
    private Response queryNestedFieldWithExpandNestedDocs(
        final String index,
        final Integer k,
        final Object[] vector,
        final String filterName,
        final String filterValue
    ) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, FIELD_NAME_NESTED);
        builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
        builder.field(VECTOR, vector);
        builder.field(K, k);
        builder.field(EXPAND_NESTED, true);
        if (filterName != null && filterValue != null) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);
            builder.field(filterName, filterValue);
            builder.endObject();
            builder.endObject();
        }

        builder.endObject().endObject().endObject();
        builder.field(INNER_HITS);
        builder.startObject().endObject();
        builder.endObject().endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }
}
