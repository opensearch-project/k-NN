/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.junit.After;
import org.opensearch.client.ResponseException;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

public class VectorDataTypeIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index-vec-dt";
    private static final String FIELD_NAME = "test-field-vec-dt";
    private static final String PROPERTIES_FIELD = "properties";
    private static final String DOC_ID = "doc1";
    private static final String TYPE_FIELD_NAME = "type";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final int EF_CONSTRUCTION = 128;
    private static final int M = 16;

    @After
    @SneakyThrows
    public final void cleanUp() {
        deleteKNNIndex(INDEX_NAME);
    }

    // Validate if we are able to create an index by setting data_type field as byte and add a doc to it
    @SneakyThrows
    public void testAddDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { 6, 6 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    // Validate by creating an index by setting data_type field as byte, add a doc to it and update it later.
    @SneakyThrows
    public void testUpdateDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { -36, 78 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        Byte[] updatedVector = { 89, -8 };
        updateKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, updatedVector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    // Validate by creating an index by setting data_type field as byte, add a doc to it and delete it later.
    @SneakyThrows
    public void testDeleteDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { 35, -46 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        deleteKnnDoc(INDEX_NAME, DOC_ID);
        refreshAllIndices();

        assertEquals(0, getDocCount(INDEX_NAME));
    }

    // Set an invalid value for data_type field while creating the index which should throw an exception
    public void testInvalidVectorDataType() {
        String vectorDataType = "invalidVectorType";
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, vectorDataType)
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s]",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES
                    )
                )
        );
    }

    // Set null value for data_type field while creating the index which should throw an exception
    public void testVectorDataTypeAsNull() {
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, null));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] on mapper [%s] of type [%s] must not have a [null] value",
                        VECTOR_DATA_TYPE_FIELD,
                        FIELD_NAME,
                        KNN_VECTOR_TYPE
                    )
                )
        );
    }

    // Create an index with byte vector data_type and add a doc with decimal values which should throw exception
    @SneakyThrows
    public void testInvalidVectorData() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Float[] vector = { -10.76f, 15.89f };

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue()
                    )
                )
        );
    }

    // Create an index with byte vector data_type and add a doc with values out of byte range which should throw exception
    @SneakyThrows
    public void testInvalidByteVectorRange() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Float[] vector = { -1000f, 155f };

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        Byte.MIN_VALUE,
                        Byte.MAX_VALUE
                    )
                )
        );
    }

    // Create an index with byte vector data_type using nmslib engine which should throw an exception
    public void testByteVectorDataTypeWithNmslibEngine() {
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndexMappingWithNmslibEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue())
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field with value [%s] is only supported for [%s] engine",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        LUCENE_NAME
                    )
                )
        );
    }

    private void createKnnIndexMappingWithNmslibEngine(int dimension, SpaceType spaceType, String vectorDataType) throws Exception {
        createKnnIndexMappingWithCustomEngine(dimension, spaceType, vectorDataType, KNNEngine.NMSLIB.getName());
    }

    private void createKnnIndexMappingWithLuceneEngine(int dimension, SpaceType spaceType, String vectorDataType) throws Exception {
        createKnnIndexMappingWithCustomEngine(dimension, spaceType, vectorDataType, KNNEngine.LUCENE.getName());
    }

    private void createKnnIndexMappingWithCustomEngine(int dimension, SpaceType spaceType, String vectorDataType, String engine)
        throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, engine)
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = Strings.toString(builder);
        createKnnIndex(INDEX_NAME, mapping);
    }
}
