/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_ENGINE;

/**
 * Integration tests for LateInteractionFieldMapper that test end-to-end functionality
 * by creating indices, mappings, and indexing documents with multi-vector fields.
 */
public class LateInteractionFieldMapperTests extends KNNSingleNodeTestCase {

    private static final String TEST_FIELD_NAME = "test-multi-vector-field";
    private static final int TEST_DIMENSION = 8;

    @SneakyThrows
    public void testIndexing_whenValidMultiVector_thenSucceed() {
        String indexName = "test-late-interaction-valid";
        createKNNIndex(indexName);

        // Create mapping with multiVector=true
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Index a document with a multi-vector (nested array of vectors)
        float[][] multiVector = new float[][] {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}
        };

        IndexResponse response = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, multiVector);
        assertEquals(RestStatus.CREATED, response.status());
    }

    @SneakyThrows
    public void testIndexing_whenMultipleMultiVectorDocuments_thenSucceed() {
        String indexName = "test-late-interaction-multiple";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Index first document with ID 1
        float[][] multiVector1 = new float[][] {
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}
        };
        IndexResponse response1 = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, multiVector1, 1);
        assertEquals(RestStatus.CREATED, response1.status());

        // Index second document with ID 2
        float[][] multiVector2 = new float[][] {
            {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f},
            {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f},
            {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f}
        };
        IndexResponse response2 = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, multiVector2, 2);
        assertEquals(RestStatus.CREATED, response2.status());
    }

    @SneakyThrows
    public void testIndexing_whenDimensionMismatch_thenFail() {
        String indexName = "test-late-interaction-dimension-mismatch";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Try to index a document with wrong dimension
        float[][] incorrectMultiVector = new float[][] {
            {1.0f, 2.0f, 3.0f}, // Only 3 dimensions instead of 8
            {4.0f, 5.0f, 6.0f}
        };

        expectThrows(
            MapperParsingException.class,
            () -> indexMultiVectorDocument(indexName, TEST_FIELD_NAME, incorrectMultiVector)
        );
    }

    @SneakyThrows
    public void testIndexing_whenSingleVectorInsteadOfMulti_thenFail() {
        String indexName = "test-late-interaction-single-vector";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Try to index a single vector instead of multi-vector
        IndexRequest request = new IndexRequest(indexName).source(
            XContentFactory.jsonBuilder()
                .startObject()
                .array(TEST_FIELD_NAME, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
                .endObject()
        ).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        expectThrows(MapperParsingException.class, () -> client().index(request).actionGet());
    }

    @SneakyThrows
    public void testIndexing_whenEmptyMultiVector_thenSucceed() {
        String indexName = "test-late-interaction-empty";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Index a document without the multi-vector field (empty)
        IndexRequest request = new IndexRequest(indexName).source(
            XContentFactory.jsonBuilder()
                .startObject()
                .field("other_field", "value")
                .endObject()
        ).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(request).actionGet();
        assertEquals(RestStatus.CREATED, response.status());
    }

    @SneakyThrows
    public void testIndexing_whenVaryingMultiVectorSizes_thenSucceed() {
        String indexName = "test-late-interaction-varying-sizes";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Document with ID 100 and 2 vectors
        float[][] smallMultiVector = new float[][] {
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}
        };
        IndexResponse response1 = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, smallMultiVector, 100);
        assertEquals(RestStatus.CREATED, response1.status());

        // Document with ID 200 and 5 vectors
        float[][] largeMultiVector = new float[][] {
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
            {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f},
            {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f},
            {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f}
        };
        IndexResponse response2 = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, largeMultiVector, 200);
        assertEquals(RestStatus.CREATED, response2.status());
    }

    @SneakyThrows
    public void testMapping_whenMultiVectorWithMethodContext_thenSucceed() {
        String indexName = "test-late-interaction-with-method";
        createKNNIndex(indexName);

        // Create mapping with method context
        PutMappingRequest request = new PutMappingRequest(indexName);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD_NAME)
            .field("type", "knn_vector")
            .field(DIMENSION, TEST_DIMENSION)
            .field(KNNConstants.MULTI_VECTOR_PARAMETER, true)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        request.source(builder);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());

        // Verify indexing works
        float[][] multiVector = new float[][] {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}
        };
        IndexResponse response = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, multiVector);
        assertEquals(RestStatus.CREATED, response.status());
    }

    @SneakyThrows
    public void testMapping_whenMultiVectorWithDocValues_thenSucceed() {
        String indexName = "test-late-interaction-doc-values";
        createKNNIndex(indexName);

        // Create mapping with doc_values enabled (default)
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Index document and verify
        float[][] multiVector = new float[][] {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}
        };
        IndexResponse response = indexMultiVectorDocument(indexName, TEST_FIELD_NAME, multiVector);
        assertEquals(RestStatus.CREATED, response.status());
    }

    @SneakyThrows
    public void testIndexing_whenDocumentWithMultipleFieldTypes_thenSucceed() {
        String indexName = "test-late-interaction-mixed-fields";
        createKNNIndex(indexName);
        createMultiVectorMapping(indexName, TEST_FIELD_NAME, TEST_DIMENSION);

        // Index document with multi-vector, ID, and additional metadata
        float[][] multiVector = new float[][] {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
            {17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}
        };

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        // Add document ID
        builder.field("id", 42);

        // Add multi-vector field
        builder.startArray(TEST_FIELD_NAME);
        for (float[] vector : multiVector) {
            builder.startArray();
            for (float value : vector) {
                builder.value(value);
            }
            builder.endArray();
        }
        builder.endArray();

        // Add various metadata fields
        builder.field("title", "Sample Document");
        builder.field("category", "test-category");
        builder.field("priority", 5);
        builder.field("active", true);
        builder.field("timestamp", System.currentTimeMillis());

        builder.endObject();

        IndexRequest request = new IndexRequest(indexName).source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        IndexResponse response = client().index(request).actionGet();
        assertEquals(RestStatus.CREATED, response.status());
    }

    /*
     * TODO: Add Lucene-level tests to verify reading indexed multi-vectors
     *
     * Future tests should validate:
     * 1. Reading BinaryDocValues from LateInteractionField to verify the encoded multi-vector format
     * 2. Using LateInteractionFloatValuesSource to compute MaxSim scores with query vectors
     * 3. Verifying dot product calculations between indexed multi-vectors and query vectors
     *
     * These tests require proper access to:
     * - IndicesService to get IndexShard
     * - Engine.Searcher to access the Lucene IndexReader
     * - LeafReaderContext for reading per-segment data
     * - BinaryDocValues API to read the encoded multi-vector data
     * - LateInteractionFloatValuesSource for computing similarity scores
     *
     * Example test scenarios:
     * - Index multi-vectors with known values and verify they can be read back correctly
     * - Compute dot products between document vectors and query vectors
     * - Validate MaxSim scoring (maximum similarity across all vector pairs)
     */

    // Helper methods

    /**
     * Creates a multi-vector field mapping
     */
    @SneakyThrows
    private void createMultiVectorMapping(String indexName, String fieldName, int dimension) {
        PutMappingRequest request = new PutMappingRequest(indexName);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field(DIMENSION, dimension)
            .field(KNNConstants.MULTI_VECTOR_PARAMETER, true)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject();

        request.source(builder);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    /**
     * Indexes a document with a multi-vector field and additional metadata fields
     */
    @SneakyThrows
    private IndexResponse indexMultiVectorDocument(String indexName, String fieldName, float[][] multiVector) {
        return indexMultiVectorDocument(indexName, fieldName, multiVector, null);
    }

    /**
     * Indexes a document with a multi-vector field, document ID, and additional metadata fields
     */
    @SneakyThrows
    private IndexResponse indexMultiVectorDocument(String indexName, String fieldName, float[][] multiVector, Integer docId) {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        // Add document ID if provided
        if (docId != null) {
            builder.field("id", docId);
        }

        // Add multi-vector field
        builder.startArray(fieldName);
        for (float[] vector : multiVector) {
            builder.startArray();
            for (float value : vector) {
                builder.value(value);
            }
            builder.endArray();
        }
        builder.endArray();

        // Add some additional metadata fields
        builder.field("timestamp", System.currentTimeMillis());
        builder.field("type", "multi-vector-document");

        builder.endObject();

        IndexRequest request = new IndexRequest(indexName).source(builder)
            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

        return client().index(request).actionGet();
    }
}
