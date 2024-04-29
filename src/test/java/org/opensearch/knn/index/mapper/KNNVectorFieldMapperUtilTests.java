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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.ByteArrayInputStream;
import java.util.Arrays;

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
        byte[] bytes = storedField.binaryValue().bytes;
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes, 0, bytes.length);
        assertArrayEquals(
            TEST_FLOAT_VECTOR,
            KNNVectorSerializerFactory.getDefaultSerializer().byteToFloatArray(byteArrayInputStream),
            0.001f
        );

        Object vector = KNNVectorFieldMapperUtil.deserializeStoredVector(storedField.binaryValue(), VectorDataType.FLOAT);
        assertTrue(vector instanceof float[]);
        assertArrayEquals(TEST_FLOAT_VECTOR, (float[]) vector, 0.001f);
    }

    public void testGetExpectedDimensionsSuccess() {
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(knnVectorFieldType.getDimension()).thenReturn(3);

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

        assertEquals(3, KNNVectorFieldMapperUtil.getExpectedDimensions(knnVectorFieldType));
        assertEquals(4, KNNVectorFieldMapperUtil.getExpectedDimensions(knnVectorFieldTypeModelBased));
    }

    public void testGetExpectedDimensionsFailure() {
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
            () -> KNNVectorFieldMapperUtil.getExpectedDimensions(knnVectorFieldTypeModelBased)
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
            () -> KNNVectorFieldMapperUtil.getExpectedDimensions(knnVectorFieldTypeModelBased)
        );
        assertEquals(String.format("Field '%s' does not have model.", fieldName), e.getMessage());
    }
}
