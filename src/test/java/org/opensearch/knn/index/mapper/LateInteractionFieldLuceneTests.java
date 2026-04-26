/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.test.hamcrest.OpenSearchAssertions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_ENGINE;

/**
 * Tests for verifying that multi-vector fields are correctly indexed as Lucene's LateInteractionField
 * and can be retrieved using BinaryDocValues.
 */
public class LateInteractionFieldLuceneTests extends KNNSingleNodeTestCase {

    private static final String TEST_INDEX_NAME = "test-late-interaction-lucene";
    private static final String TEST_FIELD_NAME = "test_multi_vector";
    private static final String DOC_ID_FIELD_NAME = "docId";
    private static final int TEST_DIMENSION = 8;

    public void testSuccess() throws IOException {
        IndexService indexService = createKNNIndex(TEST_INDEX_NAME);
        createMultiVectorMapping();

        List<float[][]> corpus = new ArrayList<>();
        int numDocs = 10;
        for (int i = 0; i < numDocs; i++) {
            corpus.add(randomMultiVector(randomIntBetween(1, 5), TEST_DIMENSION));
            indexMultiVectorDocument(i, corpus.get(i));
        }
        client().admin().indices().prepareFlush(TEST_INDEX_NAME).execute().actionGet();

        IndexShard indexShard = indexService.iterator().next();
        try (Engine.Searcher searcher = indexShard.acquireSearcher("test")) {
            IndexReader indexReader = searcher.getIndexReader();
            for (LeafReaderContext ctx : indexReader.leaves()) {
                LeafReader leafReader = ctx.reader();
                SortedNumericDocValues sortedNumericDocValues = leafReader.getSortedNumericDocValues(DOC_ID_FIELD_NAME);
                BinaryDocValues binaryDocValues = leafReader.getBinaryDocValues(TEST_FIELD_NAME);
                assertNotNull(binaryDocValues);
                while (binaryDocValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                    int id = (int) sortedNumericDocValues.nextValue();
                    BytesRef encodedBytes = binaryDocValues.binaryValue();
                    float[][] decodedMultiVector = LateInteractionField.decode(encodedBytes);
                    float[][] expectedMultiVector = corpus.get(id);

                    assertNotNull("Decoded multi-vector should not be null", decodedMultiVector);
                    assertEquals("Number of vectors should match", expectedMultiVector.length, decodedMultiVector.length);

                    for (int i = 0; i < expectedMultiVector.length; i++) {
                        assertArrayEquals("Vector " + i + " should match", expectedMultiVector[i], decodedMultiVector[i], 0.0001f);
                    }
                }
            }
        }
    }

    public void testDimensionMismatch() throws IOException {
        IndexService indexService = createKNNIndex(TEST_INDEX_NAME);
        createMultiVectorMapping();
        float[][] wrongDimMultiVector = randomMultiVector(3, TEST_DIMENSION - 1);
        float[][] incorrectMultiVector = { { 1, 2, 3, 4, 5, 6, 7, 8 }, { 2, 3 } };
        Exception e = expectThrows(Exception.class, () -> indexMultiVectorDocument(0, incorrectMultiVector));
        assertTrue(e.getCause() instanceof MapperParsingException);
        e = expectThrows(Exception.class, () -> indexMultiVectorDocument(0, wrongDimMultiVector));
        assertTrue(e.getCause() instanceof MapperParsingException);
    }

    public void testEmptyMultiVector() throws IOException {
        createKNNIndex(TEST_INDEX_NAME);
        createMultiVectorMapping();
        float[][] emptyVector = new float[0][0];
        indexMultiVectorDocument(0, emptyVector);
        client().admin().indices().prepareFlush(TEST_INDEX_NAME).execute().actionGet();

        GetResponse resp = client().get(new GetRequest(TEST_INDEX_NAME, "0")).actionGet();
        assertTrue(resp.isExists());
    }

    private void createMultiVectorMapping() throws IOException {
        PutMappingRequest request = new PutMappingRequest(TEST_INDEX_NAME);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD_NAME)
            .field("type", "knn_vector")
            .field(DIMENSION, TEST_DIMENSION)
            .field(KNNConstants.MULTI_VECTOR_PARAMETER, true)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject();

        request.source(builder);
        OpenSearchAssertions.assertAcked(client().admin().indices().putMapping(request).actionGet());
    }

    @SneakyThrows
    private void indexMultiVectorDocument(int docId, float[][] multiVector) {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();

        builder.startArray(TEST_FIELD_NAME);
        for (float[] vector : multiVector) {
            builder.startArray();
            for (float value : vector) {
                builder.value(value);
            }
            builder.endArray();
        }
        builder.endArray();
        builder.field(DOC_ID_FIELD_NAME, docId);
        builder.endObject();

        IndexRequest indexRequest = new IndexRequest().index(TEST_INDEX_NAME).id(Integer.toString(docId)).source(builder);

        IndexResponse response = client().index(indexRequest).get();
        assertEquals(RestStatus.CREATED, response.status());
    }

    private float[][] randomMultiVector(int numVectors, int dim) {
        float[][] multiVector = new float[numVectors][dim];
        for (int i = 0; i < numVectors; i++) {
            multiVector[i] = randomVector(dim);
        }
        return multiVector;
    }

    private float[] randomVector(int dim) {
        float[] v = new float[dim];
        Random random = random();
        for (int i = 0; i < dim; i++) {
            v[i] = random.nextFloat();
        }
        return v;
    }
}
