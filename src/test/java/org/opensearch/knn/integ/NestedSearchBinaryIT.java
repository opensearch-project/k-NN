/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class NestedSearchBinaryIT extends KNNRestTestCase {
    @After
    public void cleanUp() {
        try {
            deleteKNNIndex(INDEX_NAME);
        } catch (Exception e) {
            log.error(e);
        }
    }

    @SneakyThrows
    public void testNestedSearchWithFaissHnswBinary_whenKIsTwo_thenReturnTwoResults() {
        String nestedFieldName = "nested";
        createKnnBinaryIndexWithNestedField(INDEX_NAME, nestedFieldName, FIELD_NAME, 16, KNNEngine.FAISS);

        int totalDocCount = 15;
        for (byte i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(nestedFieldName)
                .addVectors(FIELD_NAME, new Byte[] { i, i }, new Byte[] { i, i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        String query = KNNJsonQueryBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(2)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(INDEX_NAME, query, 2);
        String entity = EntityUtils.toString(response.getEntity());

        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertNotEquals("14", parseIds(entity).get(1));
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
    public void testNestedSearchWithFaissHnswBinary_whenDoingExactSearch_thenReturnCorrectResults() {
        String nestedFieldName = "nested";
        String filterFieldName = "parking";
        createKnnBinaryIndexWithNestedField(INDEX_NAME, nestedFieldName, FIELD_NAME, 24, KNNEngine.FAISS);

        for (byte i = 1; i < 4; i++) {
            String doc = NestedKnnDocBuilder.create(nestedFieldName)
                .addVectors(FIELD_NAME, new Byte[] { i, i, i }, new Byte[] { i, i, i }, new Byte[] { i, i, i })
                .addTopLevelField(filterFieldName, i % 2 == 1 ? "true" : "false")
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Make it as an exact search by setting the threshold larger than size of filteredIds(6)
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f };
        String query = KNNJsonQueryBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("true")
            .build()
            .getQueryString();
        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    private void createKnnBinaryIndexWithNestedField(
        final String indexName,
        final String nestedFieldName,
        final String fieldName,
        final int dimension,
        final KNNEngine knnEngine
    ) throws Exception {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .engine(knnEngine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.BINARY.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, knnIndexMapping);
    }
}
