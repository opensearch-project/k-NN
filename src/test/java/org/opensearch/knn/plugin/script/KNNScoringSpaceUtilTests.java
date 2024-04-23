/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

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

    public void testGetExpectedDimensions() {
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

        KNNScoringSpaceUtil.initialize(modelDao);

        assertEquals(3, KNNScoringSpaceUtil.getExpectedDimensions(knnVectorFieldType));
        assertEquals(4, KNNScoringSpaceUtil.getExpectedDimensions(knnVectorFieldTypeModelBased));

        when(modelMetadata.getState()).thenReturn(ModelState.TRAINING);

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNScoringSpaceUtil.getExpectedDimensions(knnVectorFieldTypeModelBased)
        );
        assertEquals(String.format("Model ID '%s' is not created.", modelId), e.getMessage());

        when(knnVectorFieldTypeModelBased.getModelId()).thenReturn(null);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        MethodComponentContext methodComponentContext = mock(MethodComponentContext.class);
        String fieldName = "test-field";
        when(methodComponentContext.getName()).thenReturn(fieldName);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);
        when(knnVectorFieldTypeModelBased.getKnnMethodContext()).thenReturn(knnMethodContext);

        e = expectThrows(IllegalArgumentException.class, () -> KNNScoringSpaceUtil.getExpectedDimensions(knnVectorFieldTypeModelBased));
        assertEquals(String.format("Field '%s' does not have model.", fieldName), e.getMessage());
    }
}
