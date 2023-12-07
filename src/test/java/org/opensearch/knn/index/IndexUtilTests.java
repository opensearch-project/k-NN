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

import java.util.Map;
import java.util.Objects;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_EF_SEARCH;

public class IndexUtilTests extends KNNTestCase {
    public void testGetLoadParameters() {
        // Test faiss to ensure that space type gets set properly
        SpaceType spaceType1 = SpaceType.COSINESIMIL;
        KNNEngine knnEngine1 = KNNEngine.FAISS;
        String indexName = "my-test-index";

        Map<String, Object> loadParameters = getParametersAtLoading(spaceType1, knnEngine1, indexName);
        assertEquals(1, loadParameters.size());
        assertEquals(spaceType1.getValue(), loadParameters.get(SPACE_TYPE));

        // Test nmslib to ensure both space type and ef search are properly set
        SpaceType spaceType2 = SpaceType.L1;
        KNNEngine knnEngine2 = KNNEngine.NMSLIB;
        int efSearchValue = 413;

        // We use the constant for the setting here as opposed to the identifier of efSearch in nmslib jni
        Map<String, Object> indexSettings = ImmutableMap.of(KNN_ALGO_PARAM_EF_SEARCH, efSearchValue);

        // Because ef search comes from an index setting, we need to mock the long line of calls to get those
        // index settings
        Settings settings = Settings.builder().loadFromMap(indexSettings).build();
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.getSettings()).thenReturn(settings);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(anyString())).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.getMetadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);
        KNNSettings.state().setClusterService(clusterService);

        loadParameters = getParametersAtLoading(spaceType2, knnEngine2, indexName);
        assertEquals(2, loadParameters.size());
        assertEquals(spaceType2.getValue(), loadParameters.get(SPACE_TYPE));
        assertEquals(efSearchValue, loadParameters.get(HNSW_ALGO_EF_SEARCH));
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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

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

        ValidationException e = IndexUtil.validateKnnField(indexMetadata, field, dimension, modelDao);

        assert (Objects.requireNonNull(e).getMessage().matches("Validation Failed: 1: Invalid index. Index does not contain a mapping;"));
    }
}
