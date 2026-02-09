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
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

/**
 * Integration test for disk-based filtered exact search.
 */
@Log4j2
public class DiskBasedFilteredExactSearchIT extends KNNRestTestCase {

    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenFiltersMatchAllDocs_thenReturnCorrectResults() {
        for (CompressionLevel compressionLevel : List.of(
            CompressionLevel.x4,
            CompressionLevel.x8,
            CompressionLevel.x16,
            CompressionLevel.x32
        )) {
            testDiskBasedFilteredExactSearchWhenFiltersMatchAllDocs(compressionLevel);
        }
    }

    private void testDiskBasedFilteredExactSearchWhenFiltersMatchAllDocs(CompressionLevel compressionLevel) throws Exception {
        String indexName = INDEX_NAME + "_" + compressionLevel.getName();
        String filterFieldName = "color";
        final int expectResultSize = randomIntBetween(1, 3);
        final String filterValue = "red";

        createDiskBasedIndex(indexName, FIELD_NAME, 8, compressionLevel);

        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(
                indexName,
                String.valueOf(i),
                FIELD_NAME,
                new float[] { i, i, i, i, i, i, i, i },
                ImmutableMap.of(filterFieldName, filterValue)
            );
        }

        refreshIndex(indexName);
        forceMergeKnnIndex(indexName);
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

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

        Response response = searchKNNIndex(indexName, query, expectResultSize);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(expectResultSize, docIds.size());
        assertEquals(expectResultSize, parseTotalSearchHits(entity));

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenNonVectorFields_thenSucceed() {
        for (CompressionLevel compressionLevel : List.of(
            CompressionLevel.x4,
            CompressionLevel.x8,
            CompressionLevel.x16,
            CompressionLevel.x32
        )) {
            testDiskBasedFilteredExactSearchWithNonVectorFields(compressionLevel);
        }
    }

    private void testDiskBasedFilteredExactSearchWithNonVectorFields(CompressionLevel compressionLevel) throws Exception {
        String filterFieldName = "category";
        String filterValue = "electronics";
        String indexName = INDEX_NAME + "_" + compressionLevel.getName();
        createDiskBasedIndex(indexName, FIELD_NAME, 8, compressionLevel);

        for (int i = 0; i < 5; i++) {
            addKnnDocWithAttributes(
                indexName,
                String.valueOf(i),
                FIELD_NAME,
                new float[] { i, i, i, i, i, i, i, i },
                ImmutableMap.of(filterFieldName, filterValue)
            );
        }
        addKnnDocWithAttributes(
            indexName,
            String.valueOf(6),
            FIELD_NAME,
            new float[] { 6, 6, 6, 6, 6, 6, 6, 6 },
            ImmutableMap.of(filterFieldName, "books")
        );
        addNonKNNDoc(indexName, String.valueOf(7), filterFieldName, "clothes");

        flush(indexName, true);
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

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

        Response response = searchKNNIndex(indexName, query, 10);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(5, docIds.size());
        assertFalse(docIds.contains("6"));
        assertFalse(docIds.contains("7"));

        List<Double> scores = parseScores(entity);
        for (Double score : scores) {
            assertTrue("Score should be non-negative: " + score, score >= 0);
        }

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue("nonexistent")
            .rescoreEnabled(false)
            .build()
            .getQueryString();
        response = searchKNNIndex(indexName, query, 10);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(10)
            .filterFieldName(filterFieldName)
            .filterValue("clothes")
            .rescoreEnabled(false)
            .build()
            .getQueryString();
        response = searchKNNIndex(indexName, query, 10);
        assertOK(response);
        entity = EntityUtils.toString(response.getEntity());
        assertEquals(0, parseIds(entity).size());

        deleteKNNIndex(indexName);
    }

    /**
     * Test nested field with disk-based exact search.
     */
    @SneakyThrows
    public void testDiskBasedFilteredExactSearch_whenNestedField_thenSucceed() {
        for (CompressionLevel compressionLevel : List.of(
            CompressionLevel.x4,
            CompressionLevel.x8,
            CompressionLevel.x16,
            CompressionLevel.x32
        )) {
            testDiskBasedFilteredExactSearchWithCompressionLevel(compressionLevel);
        }
    }

    private void testDiskBasedFilteredExactSearchWithCompressionLevel(CompressionLevel compressionLevel) throws Exception {
        String indexName = INDEX_NAME + "_" + compressionLevel.getName();
        String nestedFieldName = "test_nested";
        String filterFieldName = "parking";
        String filterValue = "true";

        createNestedDiskBasedKnnIndex(indexName, 8, compressionLevel);

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
            addKnnDoc(indexName, String.valueOf(i), doc);
        }
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName);

        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

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

        Response response = searchKNNIndex(indexName, query, 10);
        assertOK(response);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));

        deleteKNNIndex(indexName);
    }

    private void createDiskBasedIndex(String indexName, String fieldName, int dimension, CompressionLevel compressionLevel)
        throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createNestedDiskBasedKnnIndex(String indexName, int dimension, CompressionLevel compressionLevel) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("test_nested")
            .field("type", "nested")
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }
}
