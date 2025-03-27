/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.Builder;
import lombok.Data;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.CheckedConsumer;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class DerivedSourceTestCase extends KNNRestTestCase {
    protected final int TEST_DIMENSION = 16;
    protected final int DOCS = 500;
    protected final static String NESTED_NAME = "test_nested";
    protected final static String FIELD_NAME = "test_vector";

    protected static final Settings DERIVED_ENABLED_SETTINGS = Settings.builder()
        .put("number_of_shards", 1)
        .put("number_of_replicas", 0)
        .put("index.knn", true)
        .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, true)
        .build();
    protected static final Settings DERIVED_DISABLED_SETTINGS = Settings.builder()
        .put("number_of_shards", 1)
        .put("number_of_replicas", 0)
        .put("index.knn", true)
        .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, false)
        .build();

    @Builder
    @Data
    protected static class IndexConfigContext {
        public String indexName;
        public List<String> vectorFieldNames;
        public int dimension;
        public Settings settings;
        public String mapping;
        public boolean isNested;
        public int docCount;
        public CheckedConsumer<IndexConfigContext, IOException> indexIngestor;
        public Function<IndexConfigContext, Object> updateVectorSupplier;
    }

    @SneakyThrows
    protected void prepareOriginalIndices(List<IndexConfigContext> indexConfigContexts) {
        assertEquals(6, indexConfigContexts.size());
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        createKnnIndex(derivedSourceEnabledContext.indexName, derivedSourceEnabledContext.settings, derivedSourceEnabledContext.mapping);
        createKnnIndex(derivedSourceDisabledContext.indexName, derivedSourceDisabledContext.settings, derivedSourceDisabledContext.mapping);
        derivedSourceEnabledContext.indexIngestor.accept(derivedSourceEnabledContext);
        derivedSourceDisabledContext.indexIngestor.accept(derivedSourceDisabledContext);
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            derivedSourceDisabledContext.indexName,
            derivedSourceEnabledContext.indexName
        );
        flush(derivedSourceEnabledContext.indexName, true);
        flush(derivedSourceDisabledContext.indexName, true);
    }

    @SneakyThrows
    protected void testMerging(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);

        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
        flush(derivedSourceEnabledContext.indexName, true);
        flush(derivedSourceDisabledContext.indexName, true);
    }

    public void assertDocsMatch(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    protected void testUpdate(List<IndexConfigContext> indexConfigContexts) {
        // Random variables
        int docWithVectorUpdate = DOCS - 4;
        int docWithVectorRemoval = 1;
        int docWithVectorUpdateFromAPI = 2;
        int docWithUpdateByQuery = 7;

        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        Object updateVector = derivedSourceDisabledContext.updateVectorSupplier.apply(derivedSourceDisabledContext);

        // Update via POST /<index>/_doc/<docid>
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDoc(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithVectorUpdate), fieldName, updateVector);
        }

        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDoc(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithVectorUpdate), fieldName, updateVector);
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Sets the doc to an empty doc
        setDocToEmpty(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithVectorRemoval));
        setDocToEmpty(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithVectorRemoval));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Use update API
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDocWithUpdateAPI(
                originalIndexNameDerivedSourceEnabled,
                String.valueOf(docWithVectorUpdateFromAPI),
                fieldName,
                updateVector
            );
        }
        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDocWithUpdateAPI(
                originalIndexNameDerivedSourceDisabled,
                String.valueOf(docWithVectorUpdateFromAPI),
                fieldName,
                updateVector
            );
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Update by query
        for (String fieldName : derivedSourceEnabledContext.vectorFieldNames) {
            updateKnnDocByQuery(originalIndexNameDerivedSourceEnabled, String.valueOf(docWithUpdateByQuery), fieldName, updateVector);
        }
        for (String fieldName : derivedSourceDisabledContext.vectorFieldNames) {
            updateKnnDocByQuery(originalIndexNameDerivedSourceDisabled, String.valueOf(docWithUpdateByQuery), fieldName, updateVector);
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    protected void testSearch(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;

        // Default - all fields should be there
        validateSearch(originalIndexNameDerivedSourceEnabled, derivedSourceEnabledContext.docCount, true, null, null);

        // Default - no fields should be there
        validateSearch(originalIndexNameDerivedSourceEnabled, derivedSourceEnabledContext.docCount, false, null, null);

        // Exclude all vectors
        validateSearch(
            originalIndexNameDerivedSourceEnabled,
            derivedSourceEnabledContext.docCount,
            true,
            null,
            derivedSourceEnabledContext.vectorFieldNames
        );

        // Include all vectors
        validateSearch(
            originalIndexNameDerivedSourceEnabled,
            derivedSourceEnabledContext.docCount,
            true,
            derivedSourceEnabledContext.vectorFieldNames,
            null
        );
    }

    @SneakyThrows
    protected void validateSearch(String indexName, int size, boolean isSourceEnabled, List<String> includes, List<String> excludes) {
        // TODO: We need to figure out a way to enhance validation
        QueryBuilder qb = new MatchAllQueryBuilder();
        Request request = new Request("POST", "/" + indexName + "/_search");

        request.addParameter("size", Integer.toString(size));
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field("query", qb);
        if (isSourceEnabled == false) {
            builder.field("_source", false);
        }
        if (includes != null) {
            builder.startObject("_source");
            builder.startArray("includes");
            for (String include : includes) {
                builder.value(include);
            }
            builder.endArray();
            builder.endObject();
        }
        if (excludes != null) {
            builder.startObject("_source");
            builder.startArray("excludes");
            for (String exclude : excludes) {
                builder.value(exclude);
            }
            builder.endArray();
            builder.endObject();
        }

        builder.endObject();
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        List<Object> hits = parseSearchResponseHits(responseBody);

        assertNotEquals(0, hits.size());
    }

    @SneakyThrows
    protected void testDelete(List<IndexConfigContext> indexConfigContexts) {
        int docToDelete = 8;
        int docToDeleteByQuery = 11;

        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;

        // Delete by API
        deleteKnnDoc(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDelete));
        deleteKnnDoc(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDelete));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Delete by query
        deleteKnnDocByQuery(originalIndexNameDerivedSourceEnabled, String.valueOf(docToDeleteByQuery));
        deleteKnnDocByQuery(originalIndexNameDerivedSourceDisabled, String.valueOf(docToDeleteByQuery));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    protected void testReindex(List<IndexConfigContext> indexConfigContexts) {
        IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        IndexConfigContext reindexFromEnabledToEnabledContext = indexConfigContexts.get(2);
        IndexConfigContext reindexFromEnabledToDisabledContext = indexConfigContexts.get(3);
        IndexConfigContext reindexFromDisabledToEnabledContext = indexConfigContexts.get(4);
        IndexConfigContext reindexFromDisabledToDisabledContext = indexConfigContexts.get(5);

        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        String reindexFromEnabledToEnabledIndexName = reindexFromEnabledToEnabledContext.indexName;
        String reindexFromEnabledToDisabledIndexName = reindexFromEnabledToDisabledContext.indexName;
        String reindexFromDisabledToEnabledIndexName = reindexFromDisabledToEnabledContext.indexName;
        String reindexFromDisabledToDisabledIndexName = reindexFromDisabledToDisabledContext.indexName;

        createKnnIndex(
            reindexFromEnabledToEnabledIndexName,
            reindexFromEnabledToEnabledContext.getSettings(),
            reindexFromEnabledToEnabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromEnabledToDisabledIndexName,
            reindexFromEnabledToDisabledContext.getSettings(),
            reindexFromEnabledToDisabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromDisabledToEnabledIndexName,
            reindexFromDisabledToEnabledContext.getSettings(),
            reindexFromDisabledToEnabledContext.getMapping()
        );
        createKnnIndex(
            reindexFromDisabledToDisabledIndexName,
            reindexFromDisabledToDisabledContext.getSettings(),
            reindexFromDisabledToDisabledContext.getMapping()
        );
        refreshAllIndices();
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceEnabled, reindexFromEnabledToDisabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        reindex(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToDisabledIndexName);

        // Need to forcemerge before comparison
        refreshAllIndices();
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 1);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 1);
        refreshAllIndices();
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, originalIndexNameDerivedSourceEnabled);

        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromEnabledToEnabledIndexName);
        assertIndexBigger(originalIndexNameDerivedSourceDisabled, reindexFromDisabledToEnabledIndexName);
        assertIndexBigger(reindexFromEnabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);
        assertIndexBigger(reindexFromDisabledToDisabledIndexName, originalIndexNameDerivedSourceEnabled);
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromEnabledToEnabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromDisabledToEnabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromEnabledToDisabledIndexName
        );
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            reindexFromDisabledToDisabledIndexName
        );
    }

    @SneakyThrows
    protected void assertIndexBigger(String expectedBiggerIndex, String expectedSmallerIndex) {
        if (isExhaustive()) {
            logger.info("Checking index bigger assertion because running in exhaustive mode");
            int expectedSmaller = indexSizeInBytes(expectedSmallerIndex);
            int expectedBigger = indexSizeInBytes(expectedBiggerIndex);
            assertTrue(
                "Expected smaller index " + expectedSmaller + " was bigger than the expected bigger index:" + expectedBigger,
                expectedSmaller < expectedBigger
            );
        } else {
            logger.info("Skipping index bigger assertion because not running in exhaustive mode");
        }
    }

    protected void assertDocsMatch(int docCount, String index1, String index2) {
        for (int i = 0; i < docCount; i++) {
            assertDocMatches(i + 1, index1, index2);
        }
    }

    @SneakyThrows
    protected void assertDocMatches(int docId, String index1, String index2) {
        Map<String, Object> response1 = getKnnDoc(index1, String.valueOf(docId));
        Map<String, Object> response2 = getKnnDoc(index2, String.valueOf(docId));
        assertEquals("Docs do not match: " + docId, response1, response2);
    }

    @SneakyThrows
    protected String createVectorNonNestedMappings(final int dimension, String dataType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension);
        if (dataType != null) {
            builder.field(VECTOR_DATA_TYPE_FIELD, dataType);
        }
        builder.endObject().endObject().endObject();

        return builder.toString();
    }

    @SneakyThrows
    protected String createVectorNestedMappings(final int dimension, String dataType) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(NESTED_NAME)
            .field(TYPE, "nested")
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension);
        if (dataType != null) {
            builder.field(VECTOR_DATA_TYPE_FIELD, dataType);
        }
        builder.endObject().endObject().endObject().endObject().endObject();
        return builder.toString();
    }
}
