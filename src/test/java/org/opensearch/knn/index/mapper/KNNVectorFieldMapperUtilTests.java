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
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

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
        BytesRef bytes = new BytesRef(storedField.binaryValue().bytes);
        assertArrayEquals(TEST_FLOAT_VECTOR, KNNVectorSerializerFactory.getDefaultSerializer().byteToFloatArray(bytes), 0.001f);

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
}
