/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

public class DerivedSourceTestCase extends KNNRestTestCase {

    private static final List<Pair<String, Boolean>> INDEX_PREFIX_TO_ENABLED = List.of(
        new Pair<>("original-enable-", true),
        new Pair<>("original-disable-", false),
        new Pair<>("e2e-", true),
        new Pair<>("e2d-", false),
        new Pair<>("d2e-", true),
        new Pair<>("d2d-", false)
    );

    private static final int MIN_DIMENSION = 4;
    private static final int MAX_DIMENSION = 64;
    private static final int MIN_DOCS = 100;
    private static final int MAX_DOCS = 500;

    /**
     * Testing flat, single field base case with index configuration. The test will automatically skip adding fields for
     *  random documents to ensure it works robustly. To ensure correctness, we repeat same operations against an
     *  index without derived source enabled (baseline).
     *  {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true/false
     *     },
     *     "mappings":{
     *         "properties": {
     *             "test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     *  			"test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     * 			"test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     *             "test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     *             "test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     *             "test_float_vector": {
     *                 "type": "knn_vector",
     *                 "dimension": 16,
     *                 "data_type": float
     *             },
     *             "test-text" {
     *             	"type": "text"
     *             },
     *             "test-int" {
     *             	"type": "text"
     *             },
     *         }
     *     }
     * }
     */
    protected List<DerivedSourceUtils.IndexConfigContext> getFlatIndexContexts(String testSuitePrefix, boolean addRandom, boolean addNull) {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        long consistentRandomSeed = random().nextLong();
        for (Pair<String, Boolean> index : INDEX_PREFIX_TO_ENABLED) {
            Supplier<Integer> dimensionSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DIMENSION, MAX_DIMENSION);
            Supplier<Integer> binaryDimensionSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DIMENSION, MAX_DIMENSION, 8);
            Supplier<Integer> randomDocCountSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DOCS, MAX_DOCS);
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName(testSuitePrefix, index.getFirst(), addRandom))
                .docCount(randomDocCountSupplier.get())
                .derivedEnabled(index.getSecond())
                .random(new Random(consistentRandomSeed))
                .fields(
                    List.of(
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .fieldPath("test_float_vector")
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .fieldPath("update_float_vector")
                            .isUpdate(true)
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .fieldPath("test_byte_vector")
                            .vectorDataType(VectorDataType.BYTE)
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .fieldPath("update_byte_vector")
                            .vectorDataType(VectorDataType.BYTE)
                            .dimension(dimensionSupplier.get())
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .isUpdate(true)
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .fieldPath("test_binary_vector")
                            .vectorDataType(VectorDataType.BINARY)
                            .dimension(binaryDimensionSupplier.get())
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .fieldPath("update_binary_vector")
                            .vectorDataType(VectorDataType.BINARY)
                            .dimension(binaryDimensionSupplier.get())
                            .nullProb(addNull ? DerivedSourceUtils.DEFAULT_NULL_PROB : 0)
                            .isUpdate(true)
                            .build(),
                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").build()
                    )
                )
                .build();
            indexConfigContext.init();
            indexConfigContexts.add(indexConfigContext);
        }
        return indexConfigContexts;
    }

    /**
     * Object field
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true/false
     *     },
     *     "mappings" : {
     *       "properties" : {
     *         "path_1" : {
     *           "properties" : {
     *             "test_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             },
     *             "update_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             }
     *           }
     *         },
     *         "path_2" : {
     *           "properties" : {
     *             "path_3" : {
     *               "properties" : {
     *                 "test-int" : {
     *                   "type" : "integer"
     *                 },
     *                 "test_vector" : {
     *                   "type" : "knn_vector",
     *                   "dimension" : 16
     *                 },
     *                 "update_vector" : {
     *                   "type" : "knn_vector",
     *                   "dimension" : 16
     *                 }
     *               }
     *             },
     *             "test-text" : {
     *               "type" : "text"
     *             },
     *             "test_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             },
     *             "update_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             }
     *           }
     *         },
     *         "test-int" : {
     *           "type" : "integer"
     *         },
     *         "test-text" : {
     *           "type" : "text"
     *         },
     *         "test_vector" : {
     *           "type" : "knn_vector",
     *           "dimension" : 16
     *         },
     *         "update_vector" : {
     *           "type" : "knn_vector",
     *           "dimension" : 16
     *         }
     *       }
     *     }
     *   }
     * }
     */
    protected List<DerivedSourceUtils.IndexConfigContext> getObjectIndexContexts(String testSuitePrefix, boolean addRandom) {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        long consistentRandomSeed = random().nextLong();
        for (Pair<String, Boolean> index : INDEX_PREFIX_TO_ENABLED) {
            Supplier<Integer> dimensionSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DIMENSION, MAX_DIMENSION);
            Supplier<Integer> randomDocCountSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DOCS, MAX_DOCS);
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName(testSuitePrefix, index.getFirst(), addRandom))
                .docCount(randomDocCountSupplier.get())
                .derivedEnabled(index.getSecond())
                .random(new Random(consistentRandomSeed))
                .fields(
                    List.of(
                        DerivedSourceUtils.ObjectFieldContext.builder()
                            .fieldPath("path_1")
                            .children(
                                List.of(
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("path_1.test_vector")
                                        .build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("path_1.update_vector")
                                        .isUpdate(true)
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.ObjectFieldContext.builder()
                            .fieldPath("path_2")
                            .children(
                                List.of(
                                    DerivedSourceUtils.TextFieldType.builder().fieldPath("path_2.test-text").build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("path_2.test_vector")
                                        .build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("path_2.update_vector")
                                        .isUpdate(true)
                                        .build(),
                                    DerivedSourceUtils.ObjectFieldContext.builder()
                                        .fieldPath("path_2.path_3")
                                        .children(
                                            List.of(
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .dimension(dimensionSupplier.get())
                                                    .fieldPath("path_2.path_3.test_vector")
                                                    .build(),
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .dimension(dimensionSupplier.get())
                                                    .fieldPath("path_2.path_3.update_vector")
                                                    .isUpdate(true)
                                                    .build(),
                                                DerivedSourceUtils.IntFieldType.builder().fieldPath("path_2.path_3.test-int").build()
                                            )
                                        )
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .fieldPath("test_vector")
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .fieldPath("update_vector")
                            .isUpdate(true)
                            .build(),

                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").build()
                    )
                )
                .build();
            indexConfigContext.init();
            indexConfigContexts.add(indexConfigContext);
        }

        return indexConfigContexts;
    }

    /**
     * Testing nested fields
     * {
     *     "settings": {
     *         "index.knn" true,
     *         "index.knn.derived_source.enabled": true/false
     *     },
     *     "mappings" : {
     *       "properties" : {
     *         "nested_1" : {
     *           "type" : "nested",
     *           "properties" : {
     *             "test_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             },
     *             "update_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             }
     *           }
     *         },
     *         "nested_2" : {
     *           "type" : "nested",
     *           "properties" : {
     *             "nested_3" : {
     *               "type" : "nested",
     *               "properties" : {
     *                 "test-int" : {
     *                   "type" : "integer"
     *                 },
     *                 "test_vector" : {
     *                   "type" : "knn_vector",
     *                   "dimension" : 16
     *                 },
     *                 "update_vector" : {
     *                   "type" : "knn_vector",
     *                   "dimension" : 16
     *                 }
     *               }
     *             },
     *             "test-text" : {
     *               "type" : "text"
     *             },
     *             "test_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             },
     *             "update_vector" : {
     *               "type" : "knn_vector",
     *               "dimension" : 16
     *             }
     *           }
     *         },
     *         "test-int" : {
     *           "type" : "integer"
     *         },
     *         "test-text" : {
     *           "type" : "text"
     *         },
     *         "test_vector" : {
     *           "type" : "knn_vector",
     *           "dimension" : 16
     *         },
     *         "update_vector" : {
     *           "type" : "knn_vector",
     *           "dimension" : 16
     *         }
     *       }
     *     }
     *   }
     */
    protected List<DerivedSourceUtils.IndexConfigContext> getNestedIndexContexts(String testSuitePrefix, boolean addRandom) {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        long consistentRandomSeed = random().nextLong();
        for (Pair<String, Boolean> index : INDEX_PREFIX_TO_ENABLED) {
            Supplier<Integer> dimensionSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DIMENSION, MAX_DIMENSION);
            Supplier<Integer> randomDocCountSupplier = randomIntegerSupplier(consistentRandomSeed, MIN_DOCS, MAX_DOCS);
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName(testSuitePrefix, index.getFirst(), addRandom))
                .docCount(randomDocCountSupplier.get())
                .derivedEnabled(index.getSecond())
                .random(new Random(consistentRandomSeed))
                .fields(
                    List.of(
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_1")
                            .children(
                                List.of(
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("nested_1.test_vector")
                                        .build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("nested_1.update_vector")
                                        .isUpdate(true)
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_2")
                            .children(
                                List.of(
                                    DerivedSourceUtils.TextFieldType.builder().fieldPath("nested_2.test-text").build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .dimension(dimensionSupplier.get())
                                        .fieldPath("nested_2.test_vector")
                                        .build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .fieldPath("nested_2.update_vector")
                                        .isUpdate(true)
                                        .dimension(dimensionSupplier.get())
                                        .build(),
                                    DerivedSourceUtils.NestedFieldContext.builder()
                                        .fieldPath("nested_2.nested_3")
                                        .children(
                                            List.of(
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .dimension(dimensionSupplier.get())
                                                    .fieldPath("nested_2.nested_3.test_vector")
                                                    .build(),
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .dimension(dimensionSupplier.get())
                                                    .fieldPath("nested_2.nested_3.update_vector")
                                                    .isUpdate(true)
                                                    .build(),
                                                DerivedSourceUtils.IntFieldType.builder().fieldPath("nested_2.nested_3.test-int").build()
                                            )
                                        )
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .fieldPath("test_vector")
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(dimensionSupplier.get())
                            .fieldPath("update_vector")
                            .isUpdate(true)
                            .build(),
                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").build()
                    )
                )
                .build();
            indexConfigContext.init();
            indexConfigContexts.add(indexConfigContext);
        }
        return indexConfigContexts;
    }

    @SneakyThrows
    protected void prepareOriginalIndices(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        assertTrue(1 < indexConfigContexts.size());
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        createKnnIndex(
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.getSettings(),
            derivedSourceEnabledContext.getMapping()
        );
        createKnnIndex(
            derivedSourceDisabledContext.indexName,
            derivedSourceDisabledContext.getSettings(),
            derivedSourceDisabledContext.getMapping()
        );

        for (int i = 0; i < derivedSourceDisabledContext.docCount; i++) {
            String doc1 = derivedSourceEnabledContext.buildDoc();
            String doc2 = derivedSourceDisabledContext.buildDoc();
            assertEquals(doc1, doc2);
            addKnnDoc(derivedSourceEnabledContext.getIndexName(), String.valueOf(i + 1), doc1);
            addKnnDoc(derivedSourceDisabledContext.getIndexName(), String.valueOf(i + 1), doc2);
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            derivedSourceDisabledContext.indexName,
            derivedSourceEnabledContext.indexName
        );
    }

    @SneakyThrows
    protected void testMerging(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;
        forceMergeKnnIndex(originalIndexNameDerivedSourceEnabled, 10);
        forceMergeKnnIndex(originalIndexNameDerivedSourceDisabled, 10);
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
    }

    @SneakyThrows
    protected void testUpdate(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        String originalIndexNameDerivedSourceEnabled = derivedSourceEnabledContext.indexName;
        String originalIndexNameDerivedSourceDisabled = derivedSourceDisabledContext.indexName;

        // Update via POST /<index>/_doc/<docid>
        // For this, we are just going to replace all of the docs.
        for (int i = 0; i < derivedSourceEnabledContext.docCount; i += 11) {
            addKnnDoc(derivedSourceEnabledContext.getIndexName(), String.valueOf(i + 1), derivedSourceEnabledContext.buildDoc());
            addKnnDoc(derivedSourceDisabledContext.getIndexName(), String.valueOf(i + 1), derivedSourceDisabledContext.buildDoc());
        }
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );

        // Use update API
        for (int i = 0; i < derivedSourceEnabledContext.docCount; i += 13) {
            updateUpdateAPI(
                derivedSourceEnabledContext.getIndexName(),
                String.valueOf(i + 1),
                derivedSourceEnabledContext.partialUpdateSupplier()
            );
            updateUpdateAPI(
                derivedSourceDisabledContext.getIndexName(),
                String.valueOf(i + 1),
                derivedSourceDisabledContext.partialUpdateSupplier()
            );
        }

        // Update by query
        for (int i = 0; i < derivedSourceEnabledContext.docCount; i += 17) {
            updateKnnDocByQuery(
                derivedSourceEnabledContext.getIndexName(),
                derivedSourceEnabledContext.updateByQuerySupplier(String.valueOf(i + 1))
            );
            updateKnnDocByQuery(
                derivedSourceDisabledContext.getIndexName(),
                derivedSourceDisabledContext.updateByQuerySupplier(String.valueOf(i + 1))
            );
        }

        // Sets the doc to an empty doc - make sure this is last - it can mess up update by query
        setDocToEmpty(originalIndexNameDerivedSourceEnabled, String.valueOf(1));
        setDocToEmpty(originalIndexNameDerivedSourceDisabled, String.valueOf(1));
        refreshAllIndices();
        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            originalIndexNameDerivedSourceDisabled,
            originalIndexNameDerivedSourceEnabled
        );
    }

    @SneakyThrows
    protected void testSearch(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
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
            derivedSourceEnabledContext.collectFieldNames()
        );

        // Include all vectors
        validateSearch(
            originalIndexNameDerivedSourceEnabled,
            derivedSourceEnabledContext.docCount,
            true,
            derivedSourceEnabledContext.collectFieldNames(),
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
    protected void testDelete(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        int docToDelete = 8;
        int docToDeleteByQuery = 11;

        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
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
    protected void testReindex(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        DerivedSourceUtils.IndexConfigContext reindexFromEnabledToEnabledContext = indexConfigContexts.get(2);
        DerivedSourceUtils.IndexConfigContext reindexFromEnabledToDisabledContext = indexConfigContexts.get(3);
        DerivedSourceUtils.IndexConfigContext reindexFromDisabledToEnabledContext = indexConfigContexts.get(4);
        DerivedSourceUtils.IndexConfigContext reindexFromDisabledToDisabledContext = indexConfigContexts.get(5);

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

    protected String getIndexName(String testPrefix, String indexPrefix, boolean addRandom) {
        String indexName = (testPrefix + "-" + indexPrefix + getTestName()).toLowerCase(Locale.ROOT);
        if (addRandom) {
            indexName += randomAlphaOfLength(6);
        }
        return indexName.toLowerCase(Locale.ROOT);
    }

    protected Supplier<Integer> randomIntegerSupplier(long randomSeed, int min, int max) {
        return randomIntegerSupplier(randomSeed, min, max, 1);
    }

    protected Supplier<Integer> randomIntegerSupplier(long randomSeed, int min, int max, int multipleOf) {
        Random random = new Random(randomSeed);
        return () -> {
            // Calculate how many multiples fit within the range
            int adjustedMin = (min + multipleOf - 1) / multipleOf * multipleOf;
            int adjustedMax = max / multipleOf * multipleOf;

            // Generate a random number within the adjusted range
            int randomMultiple = random.nextInt(adjustedMin / multipleOf, (adjustedMax / multipleOf) + 1) * multipleOf;

            return randomMultiple;
        };
    }
}
