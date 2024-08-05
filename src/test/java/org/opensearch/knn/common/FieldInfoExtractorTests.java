/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.apache.lucene.index.FieldInfo;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import static org.mockito.Mockito.when;

public class FieldInfoExtractorTests extends KNNTestCase {

    private static final String MODEL_ID = "model_id";

    public void testExtractVectorDataType_whenDifferentConditions_thenSuccess() {
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        MockedStatic<ModelUtil> modelUtilMockedStatic = Mockito.mockStatic(ModelUtil.class);

        // default case
        when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(null);
        when(fieldInfo.getAttribute(KNNConstants.MODEL_ID)).thenReturn(MODEL_ID);
        modelUtilMockedStatic.when(() -> ModelUtil.getModelMetadata(MODEL_ID)).thenReturn(null);
        Assert.assertEquals(VectorDataType.DEFAULT, FieldInfoExtractor.extractVectorDataType(fieldInfo));

        // VectorDataType present in fieldInfo
        when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Assert.assertEquals(VectorDataType.BINARY, FieldInfoExtractor.extractVectorDataType(fieldInfo));

        // VectorDataType present in ModelMetadata
        ModelMetadata modelMetadata = Mockito.mock(ModelMetadata.class);
        when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(null);
        modelUtilMockedStatic.when(() -> ModelUtil.getModelMetadata(MODEL_ID)).thenReturn(modelMetadata);
        when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.BYTE);
        Assert.assertEquals(VectorDataType.BYTE, FieldInfoExtractor.extractVectorDataType(fieldInfo));

        modelUtilMockedStatic.close();
    }

    public void testExtractVectorDataType() {
        FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        when(fieldInfo.getAttribute("data_type")).thenReturn(VectorDataType.BINARY.getValue());

        assertEquals(VectorDataType.BINARY, FieldInfoExtractor.extractVectorDataType(fieldInfo));
        when(fieldInfo.getAttribute("data_type")).thenReturn(null);

        when(fieldInfo.getAttribute("model_id")).thenReturn(MODEL_ID);
        try (MockedStatic<ModelUtil> modelUtilMockedStatic = Mockito.mockStatic(ModelUtil.class)) {
            ModelMetadata modelMetadata = Mockito.mock(ModelMetadata.class);
            modelUtilMockedStatic.when(() -> ModelUtil.getModelMetadata(MODEL_ID)).thenReturn(modelMetadata);
            when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.BYTE);

            assertEquals(VectorDataType.BYTE, FieldInfoExtractor.extractVectorDataType(fieldInfo));
            when(modelMetadata.getVectorDataType()).thenReturn(null);
            when(modelMetadata.getVectorDataType()).thenReturn(VectorDataType.DEFAULT);
        }

        when(fieldInfo.getAttribute("model_id")).thenReturn(null);
        assertEquals(VectorDataType.DEFAULT, FieldInfoExtractor.extractVectorDataType(fieldInfo));
    }
}
