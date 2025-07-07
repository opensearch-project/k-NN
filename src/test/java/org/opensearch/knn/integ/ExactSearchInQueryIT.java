/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import java.util.List;

@Log4j2
public class ExactSearchInQueryIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index";
    private static final String FIELD_NAME = "test-field";
    private static final int TEST_K = 3;
    private static final String TEST_EXACT_SEARCH_SPACE_TYPE = "l2";
    private static final int DIMENSION = 2;
    private static final int BINARY_DIMENSION = 16;

    @SneakyThrows
    public void testSearchWithExactSearchSpaceType_Faiss() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        for (int i = 0; i < 5; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 3f, 3f };

        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(TEST_K)
            .exactSearchSpaceType(TEST_EXACT_SEARCH_SPACE_TYPE)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, TEST_K);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(TEST_K, docIds.size());
        assertEquals(TEST_K, parseTotalSearchHits(entity));
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testSearchWithExactSearchSpaceType_FaissBinary() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", BINARY_DIMENSION)
            .field("data_type", "binary")
            .field("space_type", "hamming")
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        for (int i = 0; i < 5; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new int[] { i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Byte[] queryVector = { 3, 3 };

        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(TEST_K)
            .exactSearchSpaceType(TEST_EXACT_SEARCH_SPACE_TYPE)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, TEST_K);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(TEST_K, docIds.size());
        assertEquals(TEST_K, parseTotalSearchHits(entity));
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testSearchWithExactSearchSpaceType_Lucene() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "lucene")
            .field("space_type", "innerproduct")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        for (int i = 0; i < 5; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new float[] { i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 3f, 3f };

        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(TEST_K)
            .exactSearchSpaceType(TEST_EXACT_SEARCH_SPACE_TYPE)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, TEST_K);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(TEST_K, docIds.size());
        assertEquals(TEST_K, parseTotalSearchHits(entity));
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testSearchWithExactSearchSpaceType_LuceneBinary() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", BINARY_DIMENSION)
            .field("data_type", "binary")
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "lucene")
            .field("space_type", "hamming")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        for (int i = 0; i < 5; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new int[] { i, i });
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Byte[] queryVector = { 3, 3 };

        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(TEST_K)
            .exactSearchSpaceType(TEST_EXACT_SEARCH_SPACE_TYPE)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, TEST_K);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(TEST_K, docIds.size());
        assertEquals(TEST_K, parseTotalSearchHits(entity));
        deleteKNNIndex(INDEX_NAME);
    }
}
