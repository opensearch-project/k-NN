/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LuceneFieldStrategyTests extends KNNTestCase {

    private static final int TEST_DIMENSION = 128;

    public void testBuildFieldTypeConfigForFloatVector() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.LUCENE);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);

        FieldTypeConfig config = LuceneFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            true
        );

        assertNotNull(config.getFieldType());
        assertNotNull(config.getVectorFieldType());
        assertNull(config.getVectorTransformer());
        assertFalse(config.isUseLuceneBasedVectorField());
    }

    public void testBuildFieldTypeConfigWithoutDocValues() {
        KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
        when(mappingConfig.getDimension()).thenReturn(TEST_DIMENSION);

        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getSpaceType()).thenReturn(SpaceType.L2);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.LUCENE);

        KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);

        FieldTypeConfig config = LuceneFieldStrategy.INSTANCE.buildFieldTypeConfig(
            mappingConfig,
            methodContext,
            libraryContext,
            VectorDataType.FLOAT,
            Version.CURRENT,
            false
        );

        assertNotNull(config.getFieldType());
        assertNull(config.getVectorFieldType());
    }

    public void testCreateFloatFieldsWithDocValuesAndStored() {
        int dimension = 3;
        FieldType fieldType = KnnFloatVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);
        FieldType vectorFieldType = new FieldType();
        vectorFieldType.setDocValuesType(DocValuesType.BINARY);
        vectorFieldType.freeze();

        float[] vector = new float[] { 1.0f, 2.0f, 3.0f };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createFloatFields(
            "test_field",
            vector,
            fieldType,
            vectorFieldType,
            true,
            true,
            false
        );

        assertEquals(3, fields.size());
    }

    public void testCreateFloatFieldsWithoutDocValuesOrStored() {
        int dimension = 3;
        FieldType fieldType = KnnFloatVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);

        float[] vector = new float[] { 1.0f, 2.0f, 3.0f };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createFloatFields("test_field", vector, fieldType, null, false, false, false);

        assertEquals(1, fields.size());
    }

    public void testCreateByteFieldsWithDocValuesAndStored() {
        int dimension = 3;
        FieldType fieldType = KnnByteVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);
        FieldType vectorFieldType = new FieldType();
        vectorFieldType.setDocValuesType(DocValuesType.BINARY);
        vectorFieldType.freeze();

        byte[] vector = new byte[] { 1, 2, 3 };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createByteFields(
            "test_field",
            vector,
            fieldType,
            vectorFieldType,
            true,
            true,
            false
        );

        assertEquals(3, fields.size());
    }

    public void testCreateByteFieldsWithoutDocValuesOrStored() {
        int dimension = 3;
        FieldType fieldType = KnnByteVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);

        byte[] vector = new byte[] { 1, 2, 3 };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createByteFields("test_field", vector, fieldType, null, false, false, false);

        assertEquals(1, fields.size());
    }

    public void testCreateFloatFieldsWithDocValuesButNullVectorFieldType() {
        int dimension = 3;
        FieldType fieldType = KnnFloatVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);

        float[] vector = new float[] { 1.0f, 2.0f, 3.0f };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createFloatFields("test_field", vector, fieldType, null, false, true, false);

        assertEquals(1, fields.size());
    }

    public void testCreateByteFieldsWithDocValuesButNullVectorFieldType() {
        int dimension = 3;
        FieldType fieldType = KnnByteVectorField.createFieldType(dimension, VectorSimilarityFunction.EUCLIDEAN);

        byte[] vector = new byte[] { 1, 2, 3 };

        List<Field> fields = LuceneFieldStrategy.INSTANCE.createByteFields("test_field", vector, fieldType, null, false, true, false);

        assertEquals(1, fields.size());
    }
}
