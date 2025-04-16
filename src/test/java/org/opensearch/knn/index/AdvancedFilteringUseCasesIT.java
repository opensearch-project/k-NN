/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Assert;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;

/**
 * This class contains the IT for some advanced and tricky use-case of filters.
 * <a href="https://github.com/opensearch-project/k-NN/issues/1356">Github issue</a>
 */
public class AdvancedFilteringUseCasesIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "advanced_filtering_test_index";

    private static final String FIELD_NAME_NESTED = "test_nested";

    private static final String FIELD_NAME_VECTOR = "test_vector";

    private static final String FILTER_FIELD = "filter";

    private static final String TERM_FIELD = "term";

    private static final int k = 20;

    private static final String FIELD_NAME_METADATA = "parking";

    private static final int NUM_DOCS = 50;

    private static final int DOCUMENT_IN_RESPONSE = 10;

    private static final Float[] QUERY_VECTOR = { 5f };

    private static final List<String> enginesToTest = KNNEngine.getEnginesThatSupportsFilters()
        .stream()
        .map(KNNEngine::getName)
        .collect(Collectors.toList());

    /**
     * {
     *     "query": {
     *         "nested": {
     *             "path": "test_nested",
     *             "query": {
     *                 "knn": {
     *                     "test_nested.test_vector": {
     *                         "vector": [
     *                             3
     *                         ],
     *                         "k": 20,
     *                         "filter": {
     *                             "nested": {
     *                                 "path": "test_nested",
     *                                 "query": {
     *                                     "term": {
     *                                         "test_nested.parking": "false"
     *                                     }
     *                                 }
     *                             }
     *                         }
     *                     }
     *                 }
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFiltering_whenNestedKNNAndFilterFieldWithNestedQueries_thenSuccess() {
        setExpectRemoteBuild(true);
        for (final String engine : enginesToTest) {
            // Set up the index with nested k-nn and metadata fields
            createKnnIndex(INDEX_NAME, createNestedMappings(1, engine));
            for (int i = 1; i <= NUM_DOCS; i++) {
                // making sure that only 2 documents have valid filters
                final String metadataFieldValue = i % 2 == 0 ? "false" : "true";
                String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                    .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i + 1 })
                    .addVectorWithMetadata(FIELD_NAME_VECTOR, new Float[] { (float) i }, FIELD_NAME_METADATA, metadataFieldValue)
                    .build();
                addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
            }
            refreshIndex(INDEX_NAME);
            forceMergeKnnIndex(INDEX_NAME);

            // Build the query with both k-nn and filters as nested fields. The filter should also have a nested context
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);
            builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
            builder.field(VECTOR, QUERY_VECTOR);
            builder.field(K, k);
            builder.startObject(FILTER_FIELD);
            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);
            builder.startObject(QUERY);
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_NESTED + "." + FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject().endObject().endObject().endObject().endObject().endObject();

            validateFilterSearch(builder.toString(), engine);
            // cleanup
            deleteKNNIndex(INDEX_NAME);
        }
    }

    /**
     * {
     *     "query": {
     *         "nested": {
     *             "path": "test_nested",
     *             "query": {
     *                 "knn": {
     *                     "test_nested.test_vector": {
     *                         "vector": [
     *                             3
     *                         ],
     *                         "k": 20,
     *                         "filter": {
     *                             "term": {
     *                                 "test_nested.parking": "false"
     *                             }
     *                         }
     *                     }
     *                 }
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFiltering_whenNestedKNNAndFilterFieldWithNoNestedContextInFilterQuery_thenSuccess() {
        setExpectRemoteBuild(true);
        for (final String engine : enginesToTest) {
            // Set up the index with nested k-nn and metadata fields
            createKnnIndex(INDEX_NAME, createNestedMappings(1, engine));
            for (int i = 1; i <= NUM_DOCS; i++) {
                // making sure that only 2 documents have valid filters
                final String metadataFieldValue = i % 2 == 0 ? "false" : "true";
                String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                    .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i + 1 })
                    .addVectorWithMetadata(FIELD_NAME_VECTOR, new Float[] { (float) i }, FIELD_NAME_METADATA, metadataFieldValue)
                    .build();
                addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
            }
            refreshIndex(INDEX_NAME);
            forceMergeKnnIndex(INDEX_NAME);

            // Build the query with both k-nn and filters as nested fields but a single nested context
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);
            builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
            builder.field(VECTOR, QUERY_VECTOR);
            builder.field(K, k);
            builder.startObject(FILTER_FIELD);
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_NESTED + "." + FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();
            builder.endObject().endObject().endObject().endObject().endObject().endObject();

            validateFilterSearch(builder.toString(), engine);

            // cleanup
            deleteKNNIndex(INDEX_NAME);
        }
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
     * 							3
     * 						],
     * 						"k": 20,
     * 						"filter": {
     * 							"term": {
     * 								"parking": "false"
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
    public void testFiltering_whenNestedKNNAndNonNestedFilterFieldWithNonNestedFilterQuery_thenSuccess() {
        setExpectRemoteBuild(true);
        for (final String engine : enginesToTest) {
            // Set up the index with nested k-nn and metadata fields
            createKnnIndex(INDEX_NAME, createNestedMappings(1, engine));
            for (int i = 1; i <= NUM_DOCS; i++) {
                final String metadataFieldValue = i % 2 == 0 ? "false" : "true";
                String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                    .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i + 1 }, new Float[] { (float) i })
                    .addTopLevelField(FIELD_NAME_METADATA, metadataFieldValue)
                    .build();
                addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
            }
            refreshIndex(INDEX_NAME);
            forceMergeKnnIndex(INDEX_NAME);

            // Build the query with k-nn field as nested query and filter on the top level fields
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);
            builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
            builder.field(VECTOR, QUERY_VECTOR);
            builder.field(K, k);
            builder.startObject(FILTER_FIELD);
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();
            builder.endObject().endObject().endObject().endObject().endObject().endObject();

            validateFilterSearch(builder.toString(), engine);

            // cleanup
            deleteKNNIndex(INDEX_NAME);
        }
    }

    /**
     * {
     *     "query": {
     *         "knn": {
     *             "test_vector": {
     *                 "vector": [
     *                     3
     *                 ],
     *                 "k": 20,
     *                 "filter": {
     *                     "bool": {
     *                         "should": [
     *                             {
     *                                 "nested": {
     *                                     "path": "test_nested",
     *                                     "query": {
     *                                         "term": {
     *                                             "test_nested.parking": "false"
     *                                         }
     *                                     }
     *                                 }
     *                             }
     *                         ]
     *                     }
     *                 }
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    public void testFiltering_whenNonNestedKNNAndNestedFilterFieldWithNestedFilterQuery_thenSuccess() {
        setExpectRemoteBuild(true);
        for (final String engine : enginesToTest) {
            // Set up the index with nested k-nn and metadata fields
            createKnnIndex(INDEX_NAME, createVectorNonNestedMappings(1, engine));
            for (int i = 1; i <= NUM_DOCS; i++) {
                final String metadataFieldValue = i % 2 == 0 ? "false" : "true";
                String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                    .addMetadata(ImmutableMap.of(FIELD_NAME_METADATA, metadataFieldValue))
                    .addTopLevelField(FIELD_NAME_VECTOR, new Float[] { (float) i })
                    .build();
                addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
            }
            refreshIndex(INDEX_NAME);
            forceMergeKnnIndex(INDEX_NAME);

            // Build the query when filters are nested with nested path and k-NN field is non nested.
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
            builder.startObject(KNN).startObject(FIELD_NAME_VECTOR);
            builder.field(VECTOR, QUERY_VECTOR);
            builder.field(K, k);
            builder.startObject(FILTER_FIELD);
            builder.startObject("bool");
            builder.startArray("should");
            builder.startObject();

            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);

            builder.startObject(QUERY);
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_NESTED + "." + FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();

            builder.endObject();

            builder.endObject();
            builder.endArray();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();

            validateFilterSearch(builder.toString(), engine);
            // cleanup
            deleteKNNIndex(INDEX_NAME);
        }
    }

    /**
     * {
     * 	"query": {
     * 		"knn": {
     * 			"test_vector": {
     * 				"vector": [
     * 					5
     * 				],
     * 				"k": 20,
     * 				"filter": {
     * 					"bool": {
     * 						"must": [
     *                           {
     * 								"nested": {
     * 									"path": "test_nested",
     * 									"query": {
     * 										"term": {
     * 											"test_nested.parking": "false"
     *                                        }
     *                                    }
     *                                }
     *                            },
     *                            {
     * 								"term": {
     * 									"parking": "false"
     *                                }
     *                            }
     * 						]
     * 					}
     * 				}
     * 			}
     * 		}
     * 	}
     * }
     */
    @SneakyThrows
    public void testFiltering_whenNonNestedKNNAndNestedFilterAndNonNestedFieldWithNestedAndNonNestedFilterQuery_thenSuccess() {
        setExpectRemoteBuild(true);
        for (final String engine : enginesToTest) {
            // Set up the index with nested k-nn and metadata fields
            createKnnIndex(INDEX_NAME, createVectorNonNestedMappings(1, engine));
            for (int i = 1; i <= NUM_DOCS; i++) {
                final String metadataFieldValue = i % 2 == 0 ? "false" : "true";
                String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                    .addMetadata(ImmutableMap.of(FIELD_NAME_METADATA, metadataFieldValue))
                    .addTopLevelField(FIELD_NAME_VECTOR, new Float[] { (float) i })
                    .addTopLevelField(FIELD_NAME_METADATA, metadataFieldValue)
                    .build();
                addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
            }
            refreshIndex(INDEX_NAME);
            forceMergeKnnIndex(INDEX_NAME);

            // Build the query when filters are nested with nested path and k-NN field is non nested.
            final XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
            builder.startObject(KNN).startObject(FIELD_NAME_VECTOR);
            builder.field(VECTOR, QUERY_VECTOR);
            builder.field(K, k);
            builder.startObject(FILTER_FIELD);
            builder.startObject("bool");
            builder.startArray("must");
            builder.startObject();
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();

            builder.startObject();

            builder.startObject(TYPE_NESTED);
            builder.field(PATH, FIELD_NAME_NESTED);

            builder.startObject(QUERY);
            builder.startObject(TERM_FIELD);
            builder.field(FIELD_NAME_NESTED + "." + FIELD_NAME_METADATA, "false");
            builder.endObject();
            builder.endObject();

            builder.endObject();

            builder.endObject();
            builder.endArray();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();
            builder.endObject();

            validateFilterSearch(builder.toString(), engine);
            // cleanup
            deleteKNNIndex(INDEX_NAME);
        }
    }

    private void validateFilterSearch(final String query, final String engine) throws IOException, ParseException {
        String response = EntityUtils.toString(performSearch(INDEX_NAME, query).getEntity());
        // Validate number of documents returned as the expected number of documents
        Assert.assertEquals("For engine " + engine + ", hits: ", DOCUMENT_IN_RESPONSE, parseHits(response));
        Assert.assertEquals("For engine " + engine + ", totalSearchHits: ", k, parseTotalSearchHits(response));
        if (KNNEngine.getEngine(engine) == KNNEngine.FAISS) {
            // Update the filter threshold to 0 to ensure that we are hitting ANN Search use case for FAISS
            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 0));
            response = EntityUtils.toString(performSearch(INDEX_NAME, query).getEntity());

            // Validate number of documents returned as the expected number of documents
            Assert.assertEquals("For engine " + engine + ", hits with ANN search :", DOCUMENT_IN_RESPONSE, parseHits(response));
            Assert.assertEquals("For engine " + engine + ", totalSearchHits with ANN search :", k, parseTotalSearchHits(response));
        }
    }

    /**
     * Sample return
     * {
     *     "properties": {
     *         "test_nested": {
     *             "type": "nested",
     *             "properties": {
     *                 "test_vector": {
     *                     "type": "knn_vector",
     *                     "dimension": 1,
     *                     "method": {
     *                         "name": "hnsw",
     *                         "space_type": "l2",
     *                         "engine": "lucene"
     *                     }
     *                 }
     *             }
     *         }
     *     }
     * }
     */
    @SneakyThrows
    private String createNestedMappings(final int dimension, final String engine) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, engine)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }

    /**
     * Sample return
     * {
     *     "properties": {
     *         "test_vector": {
     *             "type": "knn_vector",
     *             "dimension": 1,
     *             "method": {
     *                 "name": "hnsw",
     *                 "space_type": "l2",
     *                 "engine": "lucene"
     *             }
     *         },
     *         "test_nested": {
     *             "type": "nested"
     *         }
     *     }
     * }
     */
    @SneakyThrows
    private String createVectorNonNestedMappings(final int dimension, final String engine) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .endObject()
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, engine)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }
}
