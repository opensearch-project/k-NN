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

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.junit.BeforeClass;
import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.jni.JNIService;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_EF_SEARCH;

public class IndexUtilTests extends KNNTestCase {

    private static MockedStatic<JNIService> jniServiceMockedStatic;
    private static final long TEST_INDEX_ADDRESS = 0;

    @BeforeClass
    public static void setUpClass() {
        jniServiceMockedStatic = mockStatic(JNIService.class);
    }

    public void testGetLoadParameters() {
        // Test faiss to ensure that space type gets set properly
        SpaceType spaceType1 = SpaceType.COSINESIMIL;
        KNNEngine knnEngine1 = KNNEngine.FAISS;
        String indexName = "my-test-index";
        VectorDataType vectorDataType1 = VectorDataType.FLOAT;

        Map<String, Object> loadParameters = getParametersAtLoading(spaceType1, knnEngine1, indexName, vectorDataType1);
        assertEquals(2, loadParameters.size());
        assertEquals(spaceType1.getValue(), loadParameters.get(SPACE_TYPE));
        assertEquals(vectorDataType1.getValue(), loadParameters.get(VECTOR_DATA_TYPE_FIELD));

        // Test nmslib to ensure both space type and ef search are properly set
        SpaceType spaceType2 = SpaceType.L1;
        KNNEngine knnEngine2 = KNNEngine.NMSLIB;
        VectorDataType vectorDataType2 = VectorDataType.BINARY;
        int efSearchValue = 413;

        // We use the constant for the setting here as opposed to the identifier of efSearch in nmslib jni
        Map<String, Object> indexSettings = ImmutableMap.of(KNN_ALGO_PARAM_EF_SEARCH, efSearchValue);

        // Because ef search comes from an index setting, we need to mock the long line of calls to get those
        // index settings
        Settings settings = Settings.builder().loadFromMap(indexSettings).build();
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.getSettings()).thenReturn(settings);
        when(indexMetadata.getCreationVersion()).thenReturn(Version.CURRENT);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(anyString())).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.getMetadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);
        KNNSettings.state().setClusterService(clusterService);

        loadParameters = getParametersAtLoading(spaceType2, knnEngine2, indexName, vectorDataType2);
        assertEquals(3, loadParameters.size());
        assertEquals(spaceType2.getValue(), loadParameters.get(SPACE_TYPE));
        assertEquals(efSearchValue, loadParameters.get(HNSW_ALGO_EF_SEARCH));
        assertEquals(vectorDataType2.getValue(), loadParameters.get(VECTOR_DATA_TYPE_FIELD));
    }

    public void testValidateKnnField_NestedField() {
        Map<String, Object> deepFieldValues = Map.of("type", "knn_vector", "dimension", 8);
        Map<String, Object> deepField = Map.of("train-field", deepFieldValues);
        Map<String, Object> deepFieldProperties = Map.of("properties", deepField);
        Map<String, Object> nest_b = Map.of("b", deepFieldProperties);
        Map<String, Object> nest_b_properties = Map.of("properties", nest_b);
        Map<String, Object> nest_a = Map.of("a", nest_b_properties);
        Map<String, Object> properties = Map.of("properties", nest_a);

        String field = "a.b.train-field";
        int dimension = 8;

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        assertNull(e);
    }

    public void testValidateKnnField_NonNestedField() {
        Map<String, Object> fieldValues = Map.of("type", "knn_vector", "dimension", 8);
        Map<String, Object> top_level_field = Map.of("top_level_field", fieldValues);
        Map<String, Object> properties = Map.of("properties", top_level_field);
        String field = "top_level_field";
        int dimension = 8;

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        assertNull(e);
    }

    public void testValidateKnnField_NonKnnField() {
        Map<String, Object> fieldValues = Map.of("type", "text");
        Map<String, Object> top_level_field = Map.of("top_level_field", fieldValues);
        Map<String, Object> properties = Map.of("properties", top_level_field);
        String field = "top_level_field";
        int dimension = 8;
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        assert Objects.requireNonNull(e).getMessage().matches("Validation Failed: 1: Field \"" + field + "\" is not of type knn_vector.;");
    }

    public void testValidateKnnField_WrongFieldPath() {
        Map<String, Object> deepFieldValues = Map.of("type", "knn_vector", "dimension", 8);
        Map<String, Object> deepField = Map.of("train-field", deepFieldValues);
        Map<String, Object> deepFieldProperties = Map.of("properties", deepField);
        Map<String, Object> nest_b = Map.of("b", deepFieldProperties);
        Map<String, Object> nest_b_properties = Map.of("properties", nest_b);
        Map<String, Object> nest_a = Map.of("a", nest_b_properties);
        Map<String, Object> properties = Map.of("properties", nest_a);
        String field = "a.train-field";
        int dimension = 8;
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        assert (Objects.requireNonNull(e).getMessage().matches("Validation Failed: 1: Field \"" + field + "\" does not exist.;"));
    }

    public void testValidateKnnField_EmptyField() {
        Map<String, Object> deepFieldValues = Map.of("type", "knn_vector", "dimension", 8);
        Map<String, Object> deepField = Map.of("train-field", deepFieldValues);
        Map<String, Object> deepFieldProperties = Map.of("properties", deepField);
        Map<String, Object> nest_b = Map.of("b", deepFieldProperties);
        Map<String, Object> nest_b_properties = Map.of("properties", nest_b);
        Map<String, Object> nest_a = Map.of("a", nest_b_properties);
        Map<String, Object> properties = Map.of("properties", nest_a);
        String field = "";
        int dimension = 8;
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        System.out.println(Objects.requireNonNull(e).getMessage());

        assert (Objects.requireNonNull(e).getMessage().matches("Validation Failed: 1: Field path is empty.;"));
    }

    public void testValidateKnnField_EmptyIndexMetadata() {
        String field = "a.b.train-field";
        int dimension = 8;
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(null);
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(modelDao.getMetadata(anyString())).thenReturn(trainingFieldModelMetadata);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, null, null);

        assert (Objects.requireNonNull(e).getMessage().matches("Validation Failed: 1: Invalid index. Index does not contain a mapping;"));
    }

    public void testIsShareableStateContainedInIndex_whenIndexNotModelBased_thenReturnFalse() {
        String modelId = null;
        KNNEngine knnEngine = KNNEngine.FAISS;
        assertFalse(IndexUtil.isSharedIndexStateRequired(knnEngine, modelId, TEST_INDEX_ADDRESS));
    }

    public void testIsShareableStateContainedInIndex_whenFaissHNSWIsUsed_thenReturnFalse() {
        jniServiceMockedStatic.when(() -> JNIService.isSharedIndexStateRequired(anyLong(), any())).thenReturn(false);
        String modelId = "test-model";
        KNNEngine knnEngine = KNNEngine.FAISS;
        assertFalse(IndexUtil.isSharedIndexStateRequired(knnEngine, modelId, TEST_INDEX_ADDRESS));
    }

    public void testIsShareableStateContainedInIndex_whenJNIIsSharedIndexStateRequiredIsTrue_thenReturnTrue() {
        jniServiceMockedStatic.when(() -> JNIService.isSharedIndexStateRequired(anyLong(), any())).thenReturn(true);
        String modelId = "test-model";
        KNNEngine knnEngine = KNNEngine.FAISS;
        assertTrue(IndexUtil.isSharedIndexStateRequired(knnEngine, modelId, TEST_INDEX_ADDRESS));
    }

    public void testIsBinaryIndex_whenBinary_thenTrue() {
        Map<String, Object> binaryIndexParams = new HashMap<>();
        binaryIndexParams.put(VECTOR_DATA_TYPE_FIELD, "binary");
        assertTrue(IndexUtil.isBinaryIndex(KNNEngine.FAISS, binaryIndexParams));
    }

    public void testIsBinaryIndex_whenNonBinary_thenFalse() {
        Map<String, Object> nonBinaryIndexParams = new HashMap<>();
        nonBinaryIndexParams.put(VECTOR_DATA_TYPE_FIELD, "byte");
        assertFalse(IndexUtil.isBinaryIndex(KNNEngine.FAISS, nonBinaryIndexParams));
    }

    public void testValidateKnnField_whenTrainModelUseDifferentVectorDataTypeFromTrainIndex_thenThrowException() {
        Map<String, Object> fieldValues = Map.of("type", "knn_vector", "dimension", 8, "data_type", "float");
        Map<String, Object> top_level_field = Map.of("top_level_field", fieldValues);
        Map<String, Object> properties = Map.of("properties", top_level_field);
        String field = "top_level_field";
        int dimension = 8;

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, VectorDataType.BINARY, null);
        System.out.println(Objects.requireNonNull(e).getMessage());

        assert Objects.requireNonNull(e)
            .getMessage()
            .matches(
                "Validation Failed: 1: Field \""
                    + field
                    + "\" has data type float, which is different from data type used in the training request: binary;"
            );
    }

    public void testValidateKnnField_whenPassByteVectorDataType_thenThrowException() {
        Map<String, Object> fieldValues = Map.of("type", "knn_vector", "dimension", 8, "data_type", "byte");
        Map<String, Object> top_level_field = Map.of("top_level_field", fieldValues);
        Map<String, Object> properties = Map.of("properties", top_level_field);
        String field = "top_level_field";
        int dimension = 8;

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao, VectorDataType.BYTE, null);

        assert Objects.requireNonNull(e)
            .getMessage()
            .matches("Validation Failed: 1: vector data type \"" + VectorDataType.BYTE.getValue() + "\" is not supported for training.;");
    }

    public void testUpdateVectorDataTypeToParameters_whenVectorDataTypeIsBinary() {
        Map<String, Object> indexParams = new HashMap<>();
        IndexUtil.updateVectorDataTypeToParameters(indexParams, VectorDataType.BINARY);
        assertEquals(VectorDataType.BINARY.getValue(), indexParams.get(VECTOR_DATA_TYPE_FIELD));
    }

    public void testValidateKnnField_whenPassBinaryVectorDataTypeAndPQEncoder_thenThrowException() {
        Map<String, Object> fieldValues = Map.of("type", "knn_vector", "dimension", 8, "data_type", "binary", "encoder", "pq");
        Map<String, Object> top_level_field = Map.of("top_level_field", fieldValues);
        Map<String, Object> properties = Map.of("properties", top_level_field);
        String field = "top_level_field";
        int dimension = 8;

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(properties);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        ModelDao modelDao = mock(ModelDao.class);
        MethodComponentContext pq = new MethodComponentContext(ENCODER_PQ, Collections.emptyMap());
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_ENCODER_PARAMETER, pq))
        );

        ValidationException e = IndexUtil.validateKnnField(
            indexMetadata,
            field,
            dimension,
            modelDao,
            VectorDataType.BINARY,
            knnMethodContext
        );

        assert Objects.requireNonNull(e)
            .getMessage()
            .matches("Validation Failed: 1: vector data type \"binary\" is not supported for pq encoder.;");
    }
}
