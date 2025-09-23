/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.generate.Documents;
import org.opensearch.knn.generate.DocumentsGenerator;
import org.opensearch.knn.generate.IndexingType;
import org.opensearch.knn.generate.SearchTestHelper;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

import static org.opensearch.knn.common.Constants.FIELD_FILTER;
import static org.opensearch.knn.common.Constants.FIELD_TERM;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.PARAM_SIZE;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;
import static org.opensearch.knn.generate.DocumentsGenerator.DIMENSIONS;
import static org.opensearch.knn.generate.DocumentsGenerator.FILTER_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.ID_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.KNN_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.MAX_VECTOR_ELEMENT_VALUE;
import static org.opensearch.knn.generate.DocumentsGenerator.MIN_VECTOR_ELEMENT_VALUE;
import static org.opensearch.knn.generate.DocumentsGenerator.NESTED_FIELD_NAME;
import static org.opensearch.knn.generate.DocumentsGenerator.SCORE_FIELD_NAME;
import static org.opensearch.knn.index.KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

@Log4j2
public abstract class AbstractMemoryOptimizedKnnSearchIT extends KNNRestTestCase {
    protected static final String INDEX_NAME = "target_index";
    protected static final int NUM_DOCUMENTS = 200;
    protected static final int TOP_K = 20;
    protected static final String EMPTY_PARAMS = "{}";
    protected static final Consumer<Settings.Builder> NO_ADDITIONAL_SETTINGS = settingsBuilder -> {};
    protected static final Consumer<Settings.Builder> NO_BUILD_HNSW = settingsBuilder -> {
        settingsBuilder.put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, -1);
    };

    protected void doTestNonNestedIndex(
        final VectorDataType dataType,
        final String methodParams,
        final boolean isRadial,
        final SpaceType spaceType,
        final Consumer<Settings.Builder> additionalSettings
    ) {
        doTestNonNestedIndex(
            dataType,
            methodParams,
            isRadial,
            spaceType,
            additionalSettings,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED
        );
    }

    protected void doTestNonNestedIndex(
        final VectorDataType dataType,
        final String methodParams,
        final boolean isRadial,
        final SpaceType spaceType,
        final Consumer<Settings.Builder> additionalSettings,
        final Mode diskMode,
        final CompressionLevel compressionLevel
    ) {
        // Make mapping schema string.
        final NonNestedNMappingSchema mapping = new NonNestedNMappingSchema().knnFieldName(KNN_FIELD_NAME)
            .dimension(DIMENSIONS)
            .dataType(dataType)
            .mode(diskMode)
            .compressionLevel(compressionLevel)
            .methodParamString(methodParams)
            .spaceType(spaceType)
            .filterFieldName(FILTER_FIELD_NAME)
            .idFieldName(ID_FIELD_NAME);

        final String mappingStr = mapping.createString();
        final Schema schema = new Schema(mappingStr, dataType, additionalSettings);

        // Start validate dense, sparse cases.
        doKnnSearchTest(spaceType, schema, IndexingType.DENSE, isRadial);
        doKnnSearchTest(spaceType, schema, IndexingType.SPARSE, isRadial);
    }

    protected void doTestNestedIndex(
        final VectorDataType dataType,
        final String methodParams,
        final SpaceType spaceType,
        final Consumer<Settings.Builder> additionalSettings
    ) {
        doTestNestedIndex(dataType, methodParams, spaceType, additionalSettings, Mode.NOT_CONFIGURED, CompressionLevel.NOT_CONFIGURED);
    }

    protected void doTestNestedIndex(
        final VectorDataType dataType,
        final String methodParams,
        final SpaceType spaceType,
        final Consumer<Settings.Builder> additionalSettings,
        final Mode diskMode,
        final CompressionLevel compressionLevel
    ) {

        // Make mapping schema string.
        final NestedMappingSchema mapping = new NestedMappingSchema().nestedFieldName(NESTED_FIELD_NAME)
            .knnFieldName(KNN_FIELD_NAME)
            .dimension(DIMENSIONS)
            .dataType(dataType)
            .mode(diskMode)
            .compressionLevel(compressionLevel)
            .methodParamString(methodParams)
            .filterFieldName(FILTER_FIELD_NAME)
            .spaceType(spaceType)
            .idFieldName(ID_FIELD_NAME);

        final String mappingStr = mapping.createString();
        final Schema schema = new Schema(mappingStr, dataType, additionalSettings);

        // Start validate dense, sparse cases.
        doKnnSearchTest(spaceType, schema, IndexingType.DENSE_NESTED, false);
        doKnnSearchTest(spaceType, schema, IndexingType.SPARSE_NESTED, false);
    }

    @SneakyThrows
    protected void doKnnSearchTest(
        final SpaceType spaceType,
        final Schema schema,
        final IndexingType indexingType,
        final boolean isRadial,
        final boolean enableMemoryOptimizedSearch
    ) {
        // Create HNSW index
        log.info("Create HNSW index, mapping=" + schema.mapping);
        createKnnHnswIndex(schema.mapping, enableMemoryOptimizedSearch, schema.additionalSettings);

        // Prepare data set
        final Documents documents = DocumentsGenerator.create(indexingType, schema.vectorDataType, NUM_DOCUMENTS).generate();

        // Index documents
        final List<String> docStrings = documents.getDocuments();
        log.info("Adding " + docStrings.size() + " documents.");
        for (int i = 0; i < docStrings.size(); ++i) {
            addKnnDoc(INDEX_NAME, Integer.toString(i), docStrings.get(i));
        }

        // Flush index
        log.info("Flushing " + INDEX_NAME);
        flushIndex(INDEX_NAME);

        // Force merge
        log.info("Force merging " + INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME, 1);

        // Do search tests
        doKnnSearchTest(documents, schema, indexingType, spaceType, isRadial, false, true);
        doKnnSearchTest(documents, schema, indexingType, spaceType, isRadial, false, false);

        doKnnSearchTest(documents, schema, indexingType, spaceType, isRadial, true, true);
        doKnnSearchTest(documents, schema, indexingType, spaceType, isRadial, true, false);

        // Delete index
        log.info("Delete index " + INDEX_NAME);
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    protected void doKnnSearchTest(
        final SpaceType spaceType,
        final Schema schema,
        final IndexingType indexingType,
        final boolean isRadial
    ) {
        doKnnSearchTest(spaceType, schema, indexingType, isRadial, true);
    }

    @SneakyThrows
    private void doKnnSearchTest(
        final Documents documents,
        final Schema schema,
        final IndexingType indexingType,
        final SpaceType spaceType,
        final boolean isRadial,
        final boolean doExhaustiveSearch,
        boolean doFiltering
    ) {
        doFiltering = doFiltering & !isRadial;
        log.info(
            "KNN Search, indexing_type="
                + indexingType
                + ", is_radial_search="
                + isRadial
                + ", space_type="
                + spaceType
                + ", doExhaustiveSearch="
                + doExhaustiveSearch
                + ", doFiltering="
                + doFiltering
        );

        // Prepare a query, and do search.
        if (schema.vectorDataType == VectorDataType.FLOAT) {
            final float[] queryVector = SearchTestHelper.generateOneSingleFloatVector(
                DIMENSIONS,
                MIN_VECTOR_ELEMENT_VALUE,
                MAX_VECTOR_ELEMENT_VALUE,
                false
            );
            final float minSimil = documents.prepareAnswerSet(
                queryVector,
                spaceType.getKnnVectorSimilarityFunction(),
                doFiltering,
                isRadial
            );
            final List<Documents.Result> results;

            if (indexingType.isNested()) {
                results = doNestedQuery(queryVector, doExhaustiveSearch, doFiltering);
            } else {
                results = doQuery(
                    queryVector,
                    doExhaustiveSearch,
                    doFiltering,
                    isRadial,
                    isRadial ? Optional.of(minSimil) : Optional.empty()
                );
            }

            documents.validateResponse(results, indexingType);
        } else if (schema.vectorDataType == VectorDataType.BYTE) {
            final byte[] queryVector = SearchTestHelper.generateOneSingleByteVector(
                DIMENSIONS,
                MIN_VECTOR_ELEMENT_VALUE,
                MAX_VECTOR_ELEMENT_VALUE
            );
            final float minSimil = documents.prepareAnswerSet(
                VectorDataType.BYTE,
                queryVector,
                spaceType.getKnnVectorSimilarityFunction(),
                doFiltering,
                isRadial
            );
            final List<Documents.Result> results;

            if (indexingType.isNested()) {
                results = doNestedQuery(SearchTestHelper.convertToIntArray(queryVector), doExhaustiveSearch, doFiltering);
            } else {
                results = doQuery(
                    SearchTestHelper.convertToIntArray(queryVector),
                    doExhaustiveSearch,
                    doFiltering,
                    isRadial,
                    isRadial ? Optional.of(minSimil) : Optional.empty()
                );
            }

            documents.validateResponse(results, indexingType);
        } else if (schema.vectorDataType == VectorDataType.BINARY) {
            final byte[] queryVector = SearchTestHelper.generateOneSingleBinaryVector(DIMENSIONS);
            final float minSimil = documents.prepareAnswerSet(
                VectorDataType.BINARY,
                queryVector,
                spaceType.getKnnVectorSimilarityFunction(),
                doFiltering,
                false
            );

            final List<Documents.Result> results;

            if (indexingType.isNested()) {
                results = doNestedQuery(SearchTestHelper.convertToIntArray(queryVector), doExhaustiveSearch, doFiltering);
            } else {
                results = doQuery(
                    SearchTestHelper.convertToIntArray(queryVector),
                    doExhaustiveSearch,
                    doFiltering,
                    isRadial,
                    isRadial ? Optional.of(minSimil) : Optional.empty()
                );
            }

            documents.validateResponse(results, indexingType);
        } else {
            throw new AssertionError();
        }

        // Off-heap is not loaded
        checkOffheapNotLoaded();
    }

    @SneakyThrows
    private void checkOffheapNotLoaded() {
        log.info("Checking whether off-heap was not used.");

        final Request request = new Request("GET", "/_plugins/_knn/stats/total_load_time");
        final Response response = client().performRequest(request);

        /**
         * Response example:
         * {
         *   "_nodes": {
         *     "total": 1,
         *     "successful": 1,
         *     "failed": 0
         *   },
         *   "cluster_name": "kdy",
         *   "nodes": {
         *     "68RvpyACSOKQtQwZK155Aw": {
         *       "total_load_time": 0
         *     }
         *   }
         * }
         */
        final String responseBody = EntityUtils.toString(response.getEntity());
        final Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        final Object loadTimeMap = ((Map<String, Object>) responseMap.get("nodes")).values().iterator().next();
        final String loadTime = ((Map<String, Object>) loadTimeMap).get("total_load_time").toString();
        assertEquals(loadTime, "0");
    }

    @SneakyThrows
    private List<Documents.Result> doQuery(
        final Object vector,
        final boolean doExhaustiveSearch,
        final boolean doFiltering,
        final boolean isRadial,
        final Optional<Float> minSimilForRadial
    ) {
        log.info("Sending a non-nested query");

        final int size = doExhaustiveSearch ? NUM_DOCUMENTS : TOP_K;
        final XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        // Size
        if (isRadial == false) {
            builder.field(PARAM_SIZE, size);
        }

        // _source
        builder.startArray("_source");
        builder.value(ID_FIELD_NAME);
        builder.value(FILTER_FIELD_NAME);
        builder.endArray();

        // Query
        builder.startObject(QUERY);

        // KNN
        builder.startObject(KNN);

        // Target field
        builder.startObject(KNN_FIELD_NAME);
        builder.field(VECTOR, vector);
        if (isRadial == false) {
            builder.field(K, size);
        } else {
            builder.field("min_score", minSimilForRadial.get());
        }

        if (doFiltering) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);

            // Filtering half documents. filter_field has either filter-0 or filter-1 in index.
            builder.field(FILTER_FIELD_NAME, "filter-0");

            builder.endObject();
            builder.endObject();
        }

        builder.endObject().endObject().endObject().endObject();

        final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        final String queryJson = builder.toString();
        request.setJsonEntity(queryJson);

        final Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<Documents.Result> results = parseSearchResponse(responseBody);
        return results;
    }

    @SneakyThrows
    private List<Documents.Result> doNestedQuery(final Object vector, final boolean doExhaustiveSearch, final boolean doFiltering) {
        log.info("Sending a nested query");

        final int size = doExhaustiveSearch ? NUM_DOCUMENTS : TOP_K;
        final XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        // Size
        builder.field(PARAM_SIZE, size);

        // _source
        builder.startArray("_source");
        builder.value(ID_FIELD_NAME);
        builder.value(FILTER_FIELD_NAME);
        builder.endArray();

        // Query
        builder.startObject(QUERY);

        // Nested
        builder.startObject(TYPE_NESTED);

        builder.field(PATH, NESTED_FIELD_NAME);

        builder.startObject(QUERY);

        // KNN
        builder.startObject(KNN);

        // Nested field
        builder.startObject(NESTED_FIELD_NAME + "." + KNN_FIELD_NAME);

        builder.field(VECTOR, vector);
        builder.field(K, size);

        if (doFiltering) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);

            // Filtering half documents. filter_field has either filter-0 or filter-1 in index.
            builder.field(FILTER_FIELD_NAME, "filter-0");

            builder.endObject();
            builder.endObject();
        }

        builder.endObject().endObject().endObject().endObject().endObject().endObject();

        final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        final String queryJson = builder.toString();
        request.setJsonEntity(queryJson);

        final Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<Documents.Result> results = parseSearchResponse(responseBody);
        return results;
    }

    @SneakyThrows
    private List<Documents.Result> parseSearchResponse(final String responseBody) {
        /**
         * Response example:
         *
         *     "hits" : [ {
         *       "_index" : "target_index",
         *       "_id" : "106",
         *       "_score" : 6.894066E-4,
         *       "_source" : {
         *         "id_field" : "id-106",
         *         "filter_field" : "filter-0"
         *       }
         *     }, {
         *       "_index" : "target_index",
         *       "_id" : "274",
         *       "_score" : 6.845179E-4,
         *       "_source" : {
         *         "id_field" : "id-274",
         *         "filter_field" : "filter-0"
         *       }
         *     }, {
         */

        final List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits")).get("hits");

        final List<Documents.Result> results = new ArrayList<>();

        for (final Object hitObj : hits) {
            final Map<String, Object> source = (Map<String, Object>) ((Map<String, Object>) hitObj).get("_source");
            final String id = source.get(ID_FIELD_NAME).toString();
            final String filterId = source.get(FILTER_FIELD_NAME).toString();
            final double score = (double) (((Map<String, Object>) hitObj).get(SCORE_FIELD_NAME));

            results.add(new Documents.Result(id, filterId, (float) score));
        }

        return results;
    }

    protected void createKnnHnswIndex(
        final String mapping,
        final boolean enableMemOptimizedSearch,
        final Consumer<Settings.Builder> additionalSettings
    ) throws IOException {
        // Settings
        final Settings.Builder settingsBuilder = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.use_compound_file", false)
            // Disable exact search at the first phase by setting large number
            // It will fallback to exact search only if the cardinality < 1, which will not happen in IT,
            // therefore, disabling exact search.
            // Note that when HNSW is skipped building, it will fallback to exact search, this one only blocks the one in the first phase.
            .put(ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 1)
            .put(KNN_INDEX, true)
            .put(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, enableMemOptimizedSearch);

        additionalSettings.accept(settingsBuilder);

        // Create a HNSW index
        createKnnIndex(INDEX_NAME, settingsBuilder.build(), mapping);
    }

    protected record Schema(String mapping, VectorDataType vectorDataType, Consumer<Settings.Builder> additionalSettings) {
    }
}
