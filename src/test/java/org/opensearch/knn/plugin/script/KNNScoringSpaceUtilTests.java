/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.NumberFieldMapper;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNScoringSpaceUtilTests extends KNNTestCase {
    public void testFieldTypeCheck() {
        assertTrue(KNNScoringSpaceUtil.isLongFieldType(new NumberFieldMapper.NumberFieldType("field", NumberFieldMapper.NumberType.LONG)));
        assertFalse(
            KNNScoringSpaceUtil.isLongFieldType(new NumberFieldMapper.NumberFieldType("field", NumberFieldMapper.NumberType.INTEGER))
        );
        assertFalse(KNNScoringSpaceUtil.isLongFieldType(new BinaryFieldMapper.BinaryFieldType("test")));

        assertTrue(KNNScoringSpaceUtil.isBinaryFieldType(new BinaryFieldMapper.BinaryFieldType("test")));
        assertFalse(
            KNNScoringSpaceUtil.isBinaryFieldType(new NumberFieldMapper.NumberFieldType("field", NumberFieldMapper.NumberType.INTEGER))
        );

        assertTrue(KNNScoringSpaceUtil.isKNNVectorFieldType(mock(KNNVectorFieldMapper.KNNVectorFieldType.class)));
        assertFalse(KNNScoringSpaceUtil.isKNNVectorFieldType(new BinaryFieldMapper.BinaryFieldType("test")));
    }

    public void testParseLongQuery() {
        int integerQueryObject = 157;
        assertEquals(Long.valueOf(integerQueryObject), KNNScoringSpaceUtil.parseToLong(integerQueryObject));

        Long longQueryObject = 10001L;
        assertEquals(longQueryObject, KNNScoringSpaceUtil.parseToLong(longQueryObject));

        String invalidQueryObject = "invalid";
        expectThrows(IllegalArgumentException.class, () -> KNNScoringSpaceUtil.parseToLong(invalidQueryObject));
    }

    public void testParseBinaryQuery() {
        String base64String = "SrtFZw==";

        /*
         * B64:         "SrtFZw=="
         * Decoded Hex: 4ABB4567
         */

        assertEquals(new BigInteger("4ABB4567", 16), KNNScoringSpaceUtil.parseToBigInteger(base64String));
    }

    public void testParseKNNVectorQuery() {
        float[] arrayFloat = new float[] { 1.0f, 2.0f, 3.0f };
        List<Double> arrayListQueryObject = new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0));

        KNNVectorFieldMapper.KNNVectorFieldType fieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);

        when(fieldType.getDimension()).thenReturn(3);
        assertArrayEquals(arrayFloat, KNNScoringSpaceUtil.parseToFloatArray(arrayListQueryObject, 3, VectorDataType.FLOAT), 0.1f);

        expectThrows(
            IllegalStateException.class,
            () -> KNNScoringSpaceUtil.parseToFloatArray(arrayListQueryObject, 4, VectorDataType.FLOAT)
        );

        String invalidObject = "invalidObject";
        expectThrows(ClassCastException.class, () -> KNNScoringSpaceUtil.parseToFloatArray(invalidObject, 3, VectorDataType.FLOAT));
    }

    public void testIsBinaryVectorDataType_whenBinary_thenReturnTrue() {
        KNNVectorFieldMapper.KNNVectorFieldType fieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(fieldType.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        assertTrue(KNNScoringSpaceUtil.isBinaryVectorDataType(fieldType));
    }

    public void testIsBinaryVectorDataType_whenNonBinary_thenReturnFalse() {
        KNNVectorFieldMapper.KNNVectorFieldType fieldType = mock(KNNVectorFieldMapper.KNNVectorFieldType.class);
        when(fieldType.getVectorDataType()).thenReturn(randomInt() % 2 == 0 ? VectorDataType.FLOAT : VectorDataType.BYTE);
        assertFalse(KNNScoringSpaceUtil.isBinaryVectorDataType(fieldType));
    }

    public void testConvertVectorToPrimitive_whenBinaryWithValidInput_thenReturnPrimitive() {
        Number number1 = mock(Number.class);
        when(number1.floatValue()).thenReturn(1f);
        Number number2 = mock(Number.class);
        when(number2.floatValue()).thenReturn(2f);
        List<Number> vector = List.of(number1, number2);
        float[] expected = new float[] { 1, 2 };
        assertArrayEquals(expected, KNNScoringSpaceUtil.convertVectorToPrimitive(vector, VectorDataType.BINARY), 0.01f);
    }

    public void testConvertVectorToPrimitive_whenBinaryWithOutOfRange_thenException() {
        Number number1 = mock(Number.class);
        when(number1.floatValue()).thenReturn(128f);
        Number number2 = mock(Number.class);
        when(number2.floatValue()).thenReturn(-129f);
        List<Number> vector = List.of(number1, number2);
        Exception e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringSpaceUtil.convertVectorToPrimitive(vector, VectorDataType.BINARY)
        );
        assertTrue(e.getMessage().contains("KNN vector values are not within in the byte range"));
    }
}
