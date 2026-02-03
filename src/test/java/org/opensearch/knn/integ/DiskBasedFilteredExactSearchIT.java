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
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.Mode;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

/**
 * Integration test for disk-based filtered exact search.
 * Tests QuantizedVectorIdsExactKNNIterator with ON_DISK mode and filters.
 */
@Log4j2
public class DiskBasedFilteredExactSearchIT extends KNNRestTestCase {

    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenFiltersMatchAllDocs_thenReturnCorrectResults() {
        String filterFieldName = "color";
        final int expectResultSize = randomIntBetween(1, 3);
        final String filterValue = "red";

        createDiskBasedIndex(INDEX_NAME, FIELD_NAME, 8);

        // Ingest 5 vector docs with the same filter value
        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(
                String.valueOf(i),
                new float[] { i, i, i, i, i, i, i, i },
                ImmutableMap.of(filterFieldName, filterValue)
            );
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Force exact search
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(expectResultSize)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .rescoreEnabled(false)
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
     * Validates filtered k-NN exact search with disk-based mode works without errors.
     */
    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenNonVectorFields_thenSucceed() {
        String filterFieldName = "category";
        String filterValue = "electronics";

        createDiskBasedIndex(INDEX_NAME, FIELD_NAME, 8);

        // Add mixed content
        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(
                String.valueOf(i),
                new float[] { i, i, i, i, i, i, i, i },
                ImmutableMap.of(filterFieldName, filterValue)
            );
        }
        addKnnDocWithAttributes(String.valueOf(6), new float[] { 6, 6, 6, 6, 6, 6, 6, 6 }, ImmutableMap.of(filterFieldName, "books"));
        addNonKNNDoc(INDEX_NAME, String.valueOf(7), filterFieldName, "clothes");

        flush(INDEX_NAME, true);

        // Force exact search
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        // Test filtered search
        Float[] queryVector = { 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .rescoreEnabled(false)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, 10);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(5, docIds.size());
        assertFalse(docIds.contains("6"));
        assertFalse(docIds.contains("7"));

        // Verify scores are valid (non-negative for Hamming distance)
        List<Double> scores = parseScores(entity);
        for (Double score : scores) {
            assertTrue("Score should be non-negative: " + score, score >= 0);
        }

        // Test filter matches 0 docs
        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue("nonexistent")
            .rescoreEnabled(false)
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 10);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        // Test filter matches only non-vector doc
        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue("clothes")
            .rescoreEnabled(false)
            .build()
            .getQueryString();
        response = searchKNNIndex(INDEX_NAME, query, 10);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());
    }

    /**
     * Test nested field with disk-based exact search.
     * Tests NestedQuantizedVectorIdsExactKNNIterator with ON_DISK mode.
     */
    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenNestedField_thenSucceed() {
        String nestedFieldName = "test_nested";
        String filterFieldName = "parking";
        String filterValue = "true";

        createKnnIndex(8, FAISS_NAME, Mode.ON_DISK);

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(nestedFieldName)
                .addVectors(
                    FIELD_NAME,
                    new Float[] { value, value, value, value, value, value, value, value },
                    new Float[] { value, value, value, value, value, value, value, value }
                )
                .addTopLevelField(filterFieldName, i % 2 == 1 ? "true" : "false")
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f, 3f, 3f, 3f, 3f, 3f };
        String query = KNNJsonQueryBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue(filterValue)
            .rescoreEnabled(false)
            .build()
            .getQueryString();

        Response response = searchKNNIndex(INDEX_NAME, query, 10);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
    }

    private void createDiskBasedIndex(String indexName, String fieldName, int dimension) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", FAISS_NAME)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createKnnIndex(int dimension, String engine, Mode mode) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("test_nested")
            .field("type", "nested")
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, mode.getName())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", engine)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(INDEX_NAME, builder.toString());
    }
}
