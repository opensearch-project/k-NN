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
import org.junit.Assert;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

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

    public void testStoredFields_whenVectorIsBinaryType_thenSucceed() {
        StoredField storedField = KNNVectorFieldMapperUtil.createStoredFieldForByteVector(TEST_FIELD_NAME, TEST_BYTE_VECTOR);
        assertEquals(TEST_FIELD_NAME, storedField.name());
        assertEquals(TEST_BYTE_VECTOR, storedField.binaryValue().bytes);
        Object vector = KNNVectorFieldMapperUtil.deserializeStoredVector(storedField.binaryValue(), VectorDataType.BINARY);
        assertTrue(vector instanceof int[]);
        int[] byteAsIntArray = new int[TEST_BYTE_VECTOR.length];
        Arrays.setAll(byteAsIntArray, i -> TEST_BYTE_VECTOR[i]);
        assertArrayEquals(byteAsIntArray, (int[]) vector);
    }

    public void testStoredFields_whenVectorIsFloatType_thenSucceed() {
        StoredField storedField = KNNVectorFieldMapperUtil.createStoredFieldForFloatVector(TEST_FIELD_NAME, TEST_FLOAT_VECTOR);
        assertEquals(TEST_FIELD_NAME, storedField.name());
        BytesRef bytes = new BytesRef(storedField.binaryValue().bytes);
        assertArrayEquals(TEST_FLOAT_VECTOR, KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.byteToFloatArray(bytes), 0.001f);

        Object vector = KNNVectorFieldMapperUtil.deserializeStoredVector(storedField.binaryValue(), VectorDataType.FLOAT);
        assertTrue(vector instanceof float[]);
        assertArrayEquals(TEST_FLOAT_VECTOR, (float[]) vector, 0.001f);
    }

    public void testGetExpectedVectorLengthSuccess() {
        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 3));
        KNNVectorFieldType knnVectorFieldTypeBinary = mock(KNNVectorFieldType.class);
        when(knnVectorFieldTypeBinary.getKnnMappingConfig()).thenReturn(
            getMappingConfigForMethodMapping(getDefaultBinaryKNNMethodContext(), 8)
        );
        when(knnVectorFieldTypeBinary.getVectorDataType()).thenReturn(VectorDataType.BINARY);

        KNNVectorFieldType knnVectorFieldTypeModelBased = mock(KNNVectorFieldType.class);
        when(knnVectorFieldTypeModelBased.getKnnMappingConfig()).thenReturn(
            getMappingConfigForMethodMapping(getDefaultBinaryKNNMethodContext(), 8)
        );
        String modelId = "test-model";
        when(knnVectorFieldTypeModelBased.getKnnMappingConfig()).thenReturn(getMappingConfigForModelMapping(modelId, 4));
        assertEquals(3, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldType));
        assertEquals(1, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeBinary));
        assertEquals(4, KNNVectorFieldMapperUtil.getExpectedVectorLength(knnVectorFieldTypeModelBased));
    }

    public void testUseLuceneKNNVectorsFormat_withDifferentInputs_thenSuccess() {
        Assert.assertFalse(KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Version.V_2_16_0));
        Assert.assertTrue(KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Version.V_2_17_0));
        Assert.assertTrue(KNNVectorFieldMapperUtil.useLuceneKNNVectorsFormat(Version.V_3_0_0));
    }

    /**
     * Test useFullFieldNameValidation method for different OpenSearch versions
     */
    public void testUseFullFieldNameValidation() {
        Assert.assertFalse(KNNVectorFieldMapperUtil.useFullFieldNameValidation(Version.V_2_16_0));
        Assert.assertTrue(KNNVectorFieldMapperUtil.useFullFieldNameValidation(Version.V_2_17_0));
        Assert.assertTrue(KNNVectorFieldMapperUtil.useFullFieldNameValidation(Version.V_2_18_0));
    }

    public void testGetEncoderName_whenNullMethodContext_thenReturnsNull() {
        assertNull(KNNVectorFieldMapperUtil.getEncoderName(null));
    }

    public void testGetEncoderName_whenEncoderPresent_thenReturnsName() {
        for (String encoder : Arrays.asList(ENCODER_FLAT, ENCODER_SQ, ENCODER_SQ)) {
            KNNMethodContext methodContext = new KNNMethodContext(
                BuiltinKNNEngine.FAISS,
                SpaceType.L2,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(encoder, Collections.emptyMap()))
                )
            );
            assertEquals(encoder, KNNVectorFieldMapperUtil.getEncoderName(methodContext));
        }
    }

    public void testGetEncoderName_whenNoEncoderParameter_thenReturnsNull() {
        KNNMethodContext methodContext = new KNNMethodContext(
            BuiltinKNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        assertNull(KNNVectorFieldMapperUtil.getEncoderName(methodContext));
    }

    public void testGetEncoderName_whenEncoderParameterIsNotMethodComponentContext_thenReturnsNull() {
        KNNMethodContext methodContext = new KNNMethodContext(
            BuiltinKNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, "not_a_method_component_context"))
        );
        assertNull(KNNVectorFieldMapperUtil.getEncoderName(methodContext));
    }
}
