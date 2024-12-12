/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class FilteredSearchBinaryIT extends KNNRestTestCase {
    private final KNNEngine engine;

    public FilteredSearchBinaryIT(KNNEngine engine) {
        this.engine = engine;
    }

    @ParametersFactory
    public static Collection<Object[]> parameters() {
        return Arrays.asList(new Object[] { KNNEngine.LUCENE }, new Object[] { KNNEngine.FAISS });
    }

    @SneakyThrows
    public void testFilteredSearchHnswBinary_whenDoingApproximateSearch_thenReturnCorrectResults() {
        validateFilteredSearchHnswBinary(INDEX_NAME, false);
    }

    @SneakyThrows
    public void testFilteredSearchHnswBinary_whenDoingExactSearch_thenReturnCorrectResults() {
        validateFilteredSearchHnswBinary(INDEX_NAME, true);
    }

    private void validateFilteredSearchHnswBinary(final String indexName, final boolean doExactSearch) throws Exception {
        String filterFieldName = "parking";
        createKnnBinaryIndex(indexName, FIELD_NAME, 24, engine);

        for (byte i = 1; i < 4; i++) {
            addKnnDocWithAttributes(
                indexName,
                Integer.toString(i),
                FIELD_NAME,
                new float[] { i, i, i },
                ImmutableMap.of(filterFieldName, i % 2 == 1 ? "true" : "false")
            );
        }
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName);

        // Set it as 0 for approximate search and 100(larger than number of filtered id) for exact search
        updateIndexSettings(
            indexName,
            Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, doExactSearch ? 100 : 0)
        );

        Float[] queryVector = { 3f, 3f, 3f };
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("true")
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, 3);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    private void createKnnBinaryIndex(final String indexName, final String fieldName, final int dimension, final KNNEngine knnEngine)
        throws Exception {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .engine(knnEngine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.BINARY.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, knnIndexMapping);
    }
}
