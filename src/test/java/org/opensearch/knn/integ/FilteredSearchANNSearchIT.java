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
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import java.util.List;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class FilteredSearchANNSearchIT extends KNNRestTestCase {
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testFilteredSearchWithFaissHnsw_whenFiltersMatchAllDocs_thenReturnCorrectResults() {
        String filterFieldName = "color";
        final int expectResultSize = randomIntBetween(1, 3);
        final String filterValue = "red";
        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), createKnnIndexMapping(FIELD_NAME, 3, METHOD_HNSW, FAISS_NAME));

        // ingest 5 vector docs into the index with the same field {"color": "red"}
        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
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

    @SneakyThrows
    public void testFilteredSearchWithNonVectorFields_whenValid_thenSucceed() {
        String filterFieldName = "category";
        String filterValue = "electronics";

        // Create mapping with both vector and non-vector fields
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 3)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", FAISS_NAME)
            .endObject()
            .endObject()
            .startObject(filterFieldName)
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        // Add mixed content
        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
        }
        addNonKNNDoc(INDEX_NAME, "6", "description", "Product description");
        deleteKnnDoc(INDEX_NAME, "0");

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Test filtered search
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
    }

    @SneakyThrows
    public void testMixedSegmentsFilteredSearch_whenValid_thenSucceed() {
        String filterFieldName = "category";
        String filterValue = "electronics";

        // Create mapping with both vector and non-vector fields
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 3)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", FAISS_NAME)
            .endObject()
            .endObject()
            .startObject(filterFieldName)
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(INDEX_NAME, getKNNDefaultIndexSettings(), mapping);

        // Add vector docs with attributes
        for (int i = 0; i < 6; i++) {
            addKnnDocWithAttributes(String.valueOf(i), new float[] { i, i, i }, ImmutableMap.of(filterFieldName, filterValue));
        }
        flush(INDEX_NAME, true);

        // Add non-vector doc (gets its own segment)
        addNonKNNDoc(INDEX_NAME, "6", "description", "Product description");
        deleteKnnDoc(INDEX_NAME, "0");
        flush(INDEX_NAME, true);

        refreshIndex(INDEX_NAME);

        // Test filtered search on mixed segments
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
    }
}
