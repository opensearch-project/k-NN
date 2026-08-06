/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.KNNCompressionRestTestCase;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import java.util.List;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class FilteredSearchANNSearchIT extends KNNCompressionRestTestCase {

    public FilteredSearchANNSearchIT(CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testFilteredSearchWithFaissHnsw_whenFiltersMatchAllDocs_thenReturnCorrectResults() {
        String filterFieldName = "color";
        final int expectResultSize = randomIntBetween(1, 3);
        final String filterValue = "red";
        createFaissHnswIndex(3, null);

        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 0));

        Float[] queryVector = { 3f, 3f, 3f };
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

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Deletes a vector doc, creating a new segment with deleted docs but no docs present.
     * Validates filtered k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testFilteredSearchWithNonVectorFields_whenValid_thenSucceed() {
        String filterFieldName = "category";
        String filterValue = "electronics";
        createFaissHnswIndex(3, filterFieldName);

        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
        }
        addNonKNNDoc(INDEX_NAME, "6", filterFieldName, "books");
        assertEquals(6, getDocCount(INDEX_NAME));
        int segmentCountBeforeDelete = getTotalSegmentCount(INDEX_NAME);
        deleteKnnDoc(INDEX_NAME, "0");
        assertEquals(5, getDocCount(INDEX_NAME));

        flush(INDEX_NAME, true);
        int segmentCountAfterDelete = getTotalSegmentCount(INDEX_NAME);
        assertTrue(segmentCountAfterDelete > segmentCountBeforeDelete);

        Float[] queryVector = { 2f, 2f, 2f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertTrue(docIds.size() > 0);

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("nonexistent")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("books")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates separate segments: one with vector docs, one with only non-vector doc.
     * Validates filtered k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testMixedSegmentsFilteredSearch_whenValid_thenSucceed() {
        String filterFieldName = "category";
        String filterValue = "electronics";
        createFaissHnswIndex(3, filterFieldName);

        for (int i = 0; i < 6; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
        }
        flush(INDEX_NAME, true);

        addNonKNNDoc(INDEX_NAME, "6", filterFieldName, "books");
        flush(INDEX_NAME, true);

        refreshIndex(INDEX_NAME);
        int segmentCount = getTotalSegmentCount(INDEX_NAME);
        assertTrue(segmentCount >= 2);

        Float[] queryVector = { 2f, 2f, 2f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertTrue(docIds.size() > 0);

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("nonexistent")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("books")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 3);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates a doc with vector field, then updates it to remove the vector field.
     * Validates filtered k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testVectorFieldRemovalByUpdate_whenValid_thenSucceed() {
        String filterFieldName = "category";
        createFaissHnswIndex(3, filterFieldName);

        String docId = "0";
        addKnnDocWithAttributes(docId, new float[] { 1f, 1f, 1f }, ImmutableMap.of(filterFieldName, "electronics"));

        addNonKNNDoc(INDEX_NAME, docId, filterFieldName, "books");

        flush(INDEX_NAME, true);

        assertEquals(1, getDocCount(INDEX_NAME));
        Float[] queryVector = { 1f, 1f, 1f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(5)
            .filterFieldName(filterFieldName)
            .filterValue("electronics")
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, 5);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(5)
            .filterFieldName(filterFieldName)
            .filterValue("nonexistent")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 5);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(5)
            .filterFieldName(filterFieldName)
            .filterValue("books")
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 5);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());
    }

    @SneakyThrows
    private void createFaissHnswIndex(int dimension, String keywordFilterField) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension);
        addCompressionMappingFields(builder);
        builder.startObject("method").field("name", METHOD_HNSW).field("engine", FAISS_NAME).endObject().endObject();

        if (keywordFilterField != null) {
            builder.startObject(keywordFilterField)
                .field("type", "keyword")
                .endObject()
                .startObject("description")
                .field("type", "text")
                .endObject();
        }

        builder.endObject().endObject();
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), builder.toString());
    }
}
