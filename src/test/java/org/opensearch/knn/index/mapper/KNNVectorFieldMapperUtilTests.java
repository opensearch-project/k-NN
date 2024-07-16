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

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.StoredField;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.util.Arrays;
import java.util.Collections;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNVectorFieldMapperUtilTests extends KNNTestCase {

    private static final String TEST_FIELD_NAME = "test_field_name";
    private static final byte[] TEST_BYTE_VECTOR = new byte[] { -128, 0, 1, 127 };
    private static final float[] TEST_FLOAT_VECTOR = new float[] { -100.0f, 100.0f, 0f, 1f };

    public void testStoredFields_whenVectorIsByteType_thenSucceed() {
        StoredField storedField = KNNVectorFieldMapperUtil.createStoredFieldForByteVector(TEST_FIELD_NAME, TEST_BYTE_VECTOR);
        assertEquals(TEST_FIELD_NAME, storedField.name());
        assertEquals(TEST_BYTE_VECTOR, storedField.binaryValue().bytes);
        Object vector = KNNVectorFieldMapperUtil.deserializeStoredVector(storedField.binaryValue(), VectorDataType.BYTE);
        assertTrue(vector instanceof int[]);
        int[] byteAsIntArray = new int[TEST_BYTE_VECTOR.length];
        Arrays.setAll(byteAsIntArray, i -> TEST_BYTE_VECTOR[i]);
        assertArrayEquals(byteAsIntArray, (int[]) vector);
    }

    public void testStoredFields_whenVectorIsFloatType_thenSucceed() {
        StoredField storedField = KNNVectorFieldMapperUtil.createStoredFieldForFloatVector(TEST_FIELD_NAME, TEST_FLOAT_VECTOR);
        assertEquals(TEST_FIELD_NAME, storedField.name());
        BytesRef bytes = new BytesRef(storedField.binaryValue().bytes);
        assertArrayEquals(TEST_FLOAT_VECTOR, KNNVectorSerializerFactory.getDefaultSerializer().byteToFloatArray(bytes), 0.001f);

        Object vector = KNNVectorFieldMapperUtil.deserializeStoredVector(storedField.binaryValue(), VectorDataType.FLOAT);
        assertTrue(vector instanceof float[]);
        assertArrayEquals(TEST_FLOAT_VECTOR, (float[]) vector, 0.001f);
    }

    public void testGetExpectedVectorLengthSuccess() {
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldType.getDimension()).thenReturn(3);

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getDimension()).thenReturn(8);
        when(knnVectorFieldTypeBinary.getVectorDataType()).thenReturn(VectorDataType.BINARY);

        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldTypeModelBased = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldTypeModelBased.getDimension()).thenReturn(-1);
        String modelId = "test-model";
        when(knnVectorFieldTypeModelBased.getModelId()).thenReturn(modelId);

        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getState()).thenReturn(ModelState.CREATED);
        when(modelMetadata.getDimension()).thenReturn(4);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);

        KNNVectorFieldMapperUtil.initialize(modelDao);

        assertEquals(3, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType));
        assertEquals(1, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeBinary));
        assertEquals(4, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeModelBased));
    }

    public void testGetExpectedVectorLengthFailure() {
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldTypeModelBased = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldTypeModelBased.getDimension()).thenReturn(-1);
        String modelId = "test-model";
        when(knnVectorFieldTypeModelBased.getModelId()).thenReturn(modelId);

        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = mock(ModelMetadata.class);
        when(modelMetadata.getState()).thenReturn(ModelState.TRAINING);
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);

        KNNVectorFieldMapperUtil.initialize(modelDao);

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeModelBased)
        );
        assertEquals(String.format("Model ID '%s' is not created.", modelId), e.getMessage());

        when(knnVectorFieldTypeModelBased.getModelId()).thenReturn(null);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        MethodComponentContext methodComponentContext = mock(MethodComponentContext.class);
        String fieldName = "test-field";
        when(methodComponentContext.getName()).thenReturn(fieldName);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);
        when(knnVectorFieldTypeModelBased.getKnnMethodContext()).thenReturn(knnMethodContext);

        e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeModelBased)
        );
        assertEquals(String.format("Field '%s' does not have model.", fieldName), e.getMessage());
    }

    public void testValidateVectorDataType_whenBinaryFaissHNSW_thenValid() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.BINARY, null);
    }

    public void testValidateVectorDataType_whenBinaryNonFaiss_thenException() {
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_HNSW, VectorDataType.BINARY, "only supported");
        validateValidateVectorDataType(KNNEngine.NMSLIB, KNNConstants.METHOD_HNSW, VectorDataType.BINARY, "only supported");
    }

    public void testValidateVectorDataType_whenBinaryFaissIVF_thenException() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_IVF, VectorDataType.BINARY, "only supported");
    }

    public void testValidateVectorDataType_whenByteLucene_thenValid() {
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_HNSW, VectorDataType.BYTE, null);
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_IVF, VectorDataType.BYTE, null);
    }

    public void testValidateVectorDataType_whenByteNonLucene_thenException() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.BYTE, "only supported");
        validateValidateVectorDataType(KNNEngine.NMSLIB, KNNConstants.METHOD_IVF, VectorDataType.BYTE, "only supported");
    }

    public void testValidateVectorDataType_whenFloat_thenValid() {
        validateValidateVectorDataType(KNNEngine.FAISS, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, null);
        validateValidateVectorDataType(KNNEngine.LUCENE, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, null);
        validateValidateVectorDataType(KNNEngine.NMSLIB, KNNConstants.METHOD_HNSW, VectorDataType.FLOAT, null);
    }

    private void validateValidateVectorDataType(
        final KNNEngine knnEngine,
        final String methodName,
        final VectorDataType vectorDataType,
        final String expectedErrMsg
    ) {
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodName, Collections.emptyMap());
        KNNMethodContext methodContext = new KNNMethodContext(knnEngine, SpaceType.UNDEFINED, methodComponentContext);
        if (expectedErrMsg == null) {
            KNNVectorFieldMapperUtil.validateVectorDataType(methodContext, vectorDataType);
        } else {
            Exception ex = expectThrows(
                IllegalArgumentException.class,
                () -> KNNVectorFieldMapperUtil.validateVectorDataType(methodContext, vectorDataType)
            );
            assertTrue(ex.getMessage().contains(expectedErrMsg));
        }
    }
}
