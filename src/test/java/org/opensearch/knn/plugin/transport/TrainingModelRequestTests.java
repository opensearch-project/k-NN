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

package org.opensearch.knn.plugin.transport;

import com.google.common.collect.ImmutableMap;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class TrainingModelRequestTests extends KNNTestCase {

    public void testStreams() throws IOException {
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";
        String preferredNode = "test-preferred-node";
        String description = "some test description";

        TrainingModelRequest original1 = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            preferredNode,
            description,
            VectorDataType.DEFAULT
        );

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        original1.writeTo(streamOutput);
        TrainingModelRequest copy1 = new TrainingModelRequest(streamOutput.bytes().streamInput());

        assertEquals(original1.getModelId(), copy1.getModelId());
        assertEquals(original1.getKnnMethodContext(), copy1.getKnnMethodContext());
        assertEquals(original1.getDimension(), copy1.getDimension());
        assertEquals(original1.getTrainingIndex(), copy1.getTrainingIndex());
        assertEquals(original1.getTrainingField(), copy1.getTrainingField());
        assertEquals(original1.getPreferredNodeId(), copy1.getPreferredNodeId());
        assertEquals(original1.getVectorDataType(), copy1.getVectorDataType());

        // Also, check when preferred node and model id and description are null
        TrainingModelRequest original2 = new TrainingModelRequest(
            null,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        streamOutput = new BytesStreamOutput();
        original2.writeTo(streamOutput);
        TrainingModelRequest copy2 = new TrainingModelRequest(streamOutput.bytes().streamInput());

        assertEquals(original2.getModelId(), copy2.getModelId());
        assertEquals(original2.getKnnMethodContext(), copy2.getKnnMethodContext());
        assertEquals(original2.getDimension(), copy2.getDimension());
        assertEquals(original2.getTrainingIndex(), copy2.getTrainingIndex());
        assertEquals(original2.getTrainingField(), copy2.getTrainingField());
        assertEquals(original2.getPreferredNodeId(), copy2.getPreferredNodeId());
        assertEquals(original2.getVectorDataType(), copy2.getVectorDataType());
    }

    public void testGetters() {
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = KNNMethodContext.getDefault();
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";
        String preferredNode = "test-preferred-node";
        String description = "some test description";
        int maxVectorCount = 100;
        int searchSize = 101;
        int trainingSetSizeInKB = 102;

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            preferredNode,
            description,
            VectorDataType.DEFAULT
        );

        trainingModelRequest.setMaximumVectorCount(maxVectorCount);
        trainingModelRequest.setSearchSize(searchSize);
        trainingModelRequest.setTrainingDataSizeInKB(trainingSetSizeInKB);

        assertEquals(modelId, trainingModelRequest.getModelId());
        assertEquals(knnMethodContext, trainingModelRequest.getKnnMethodContext());
        assertEquals(dimension, trainingModelRequest.getDimension());
        assertEquals(trainingIndex, trainingModelRequest.getTrainingIndex());
        assertEquals(trainingField, trainingModelRequest.getTrainingField());
        assertEquals(preferredNode, trainingModelRequest.getPreferredNodeId());
        assertEquals(description, trainingModelRequest.getDescription());
        assertEquals(maxVectorCount, trainingModelRequest.getMaximumVectorCount());
        assertEquals(searchSize, trainingModelRequest.getSearchSize());
        assertEquals(trainingSetSizeInKB, trainingModelRequest.getTrainingDataSizeInKB());
    }

    public void testValidation_invalid_modelIdAlreadyExists() {
        // Check that validation produces exception when the modelId passed in already has a model
        // associated with it

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return metadata for modelId to recognize it is a duplicate
        ModelDao modelDao = mock(ModelDao.class);
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );
        when(modelDao.getMetadata(modelId)).thenReturn(modelMetadata);

        // ModelId is not added to model graveyard
        when(modelDao.isModelInGraveyard(modelId)).thenReturn(false);

        // This cluster service will result in no validation exceptions
        ClusterService clusterService = getClusterServiceForValidReturns(trainingIndex, trainingField, dimension);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("already exists"));
    }

    // Check that the validation produces an exception when we are
    // training a model with modelId that is in model graveyard
    public void testValidation_blocked_modelId() {

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return true to recognize that the modelId is in graveyard
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.isModelInGraveyard(modelId)).thenReturn(true);

        // This cluster service will result in no validation exceptions
        ClusterService clusterService = getClusterServiceForValidReturns(trainingIndex, trainingField, dimension);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces an error message that modelId is being deleted
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("is being deleted"));
    }

    public void testValidation_invalid_invalidMethodContext() {
        // Check that validation produces exception when the method is invalid and does not require training

        // Setup the training request
        String modelId = "test-model-id";

        // Mock throwing an exception on validation
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        String validationExceptionMessage = "knn method invalid";
        ValidationException validationException = new ValidationException();
        validationException.addValidationError(validationExceptionMessage);
        when(knnMethodContext.validate()).thenReturn(validationException);

        when(knnMethodContext.isTrainingRequired()).thenReturn(false);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return null so that no exception is produced
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // This cluster service will result in no validation exceptions
        ClusterService clusterService = getClusterServiceForValidReturns(trainingIndex, trainingField, dimension);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(2, validationErrors.size());
        assertTrue(validationErrors.get(0).contains(validationExceptionMessage));
        assertTrue(validationErrors.get(1).contains("Method does not require training."));
    }

    public void testValidation_invalid_trainingIndexDoesNotExist() {
        // Check that validation produces exception when the training index doesnt exist

        // Setup the training request
        String modelId = "test-model-id";

        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);

        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return null so that no exception is produced
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(null);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("does not exist"));
    }

    public void testValidation_invalid_trainingFieldDoesNotExist() {
        // Check that validation produces exception when the training field doesnt exist

        // Setup the training request
        String modelId = "test-model-id";

        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);

        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return null so that no exception is produced
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // Return empty mapping so that training field does not exist
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(Collections.emptyMap());
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("does not exist"));
    }

    public void testValidation_invalid_trainingFieldNotKnnVector() {
        // Check that validation produces exception when field is not knn_vector

        // Setup the training request
        String modelId = "test-model-id";

        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);

        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return null so that no exception is produced
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // Return mapping with different type
        Map<String, Object> mappingMap = ImmutableMap.of(
            "properties",
            ImmutableMap.of(trainingField, ImmutableMap.of("type", "int", KNNConstants.DIMENSION, dimension))
        );
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(mappingMap);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("not of type"));
    }

    public void testValidation_invalid_dimensionDoesNotMatch() {
        // Check that validation produces exception when dimension does not match

        // Setup the training request
        String modelId = "test-model-id";

        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);

        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return null so that no exception is produced
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // Return mapping with different dimension
        Map<String, Object> mappingMap = ImmutableMap.of(
            "properties",
            ImmutableMap.of(
                trainingField,
                ImmutableMap.of("type", KNNVectorFieldMapper.CONTENT_TYPE, KNNConstants.DIMENSION, dimension + 1)
            )
        );
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(mappingMap);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("different from dimension"));
    }

    public void testValidation_invalid_preferredNodeDoesNotExist() {
        // Check that validation produces exception preferred node does not exist

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";
        String preferredNode = "preferred-node";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            preferredNode,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return metadata for modelId to recognize it is a duplicate
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // This cluster service mocking should not produce exception
        Map<String, Object> mappingMap = ImmutableMap.of(
            "properties",
            ImmutableMap.of(trainingField, ImmutableMap.of("type", KNNVectorFieldMapper.CONTENT_TYPE, KNNConstants.DIMENSION, dimension))
        );

        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(mappingMap);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);

        // Empty set of data nodes to produce exception
        DiscoveryNodes discoveryNodes = mock(DiscoveryNodes.class);
        when(discoveryNodes.getDataNodes()).thenReturn(Map.of());

        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        when(clusterState.nodes()).thenReturn(discoveryNodes);

        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("Preferred node"));
    }

    public void testValidation_invalid_descriptionToLong() {

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";
        String trainingFieldModeId = "training-field-model-id";

        char[] chars = new char[KNNConstants.MAX_MODEL_DESCRIPTION_LENGTH + 1];
        Arrays.fill(chars, 'a');
        String description = new String(chars);

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            description,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return metadata for modelId to recognize it is a duplicate
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);
        when(modelDao.getMetadata(trainingFieldModeId)).thenReturn(trainingFieldModelMetadata);

        // Cluster service that wont produce validation exception
        ClusterService clusterService = getClusterServiceForValidReturns(trainingIndex, trainingField, dimension);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNotNull(exception);
        List<String> validationErrors = exception.validationErrors();
        assertEquals(1, validationErrors.size());
        assertTrue(validationErrors.get(0).contains("Description exceeds limit"));
    }

    public void testValidation_valid_trainingIndexBuiltFromMethod() {
        // This cluster service will result in no validation exceptions

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return metadata for modelId to recognize it is a duplicate
        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);

        // Mock the cluster service to not produce exceptions
        ClusterService clusterService = getClusterServiceForValidReturns(trainingIndex, trainingField, dimension);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNull(exception);
    }

    public void testValidation_valid_trainingIndexBuiltFromModel() {

        // Setup the training request
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.validate()).thenReturn(null);
        when(knnMethodContext.isTrainingRequired()).thenReturn(true);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(MethodComponentContext.EMPTY);
        int dimension = 10;
        String trainingIndex = "test-training-index";
        String trainingField = "test-training-field";
        String trainingFieldModeId = "training-field-model-id";

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            null,
            null,
            VectorDataType.DEFAULT
        );

        // Mock the model dao to return metadata for modelId to recognize it is a duplicate
        ModelMetadata trainingFieldModelMetadata = mock(ModelMetadata.class);
        when(trainingFieldModelMetadata.getDimension()).thenReturn(dimension);
        when(trainingFieldModelMetadata.getState()).thenReturn(ModelState.CREATED);

        ModelDao modelDao = mock(ModelDao.class);
        when(modelDao.getMetadata(modelId)).thenReturn(null);
        when(modelDao.getMetadata(trainingFieldModeId)).thenReturn(trainingFieldModelMetadata);

        // Return model id instead of dimension directly
        Map<String, Object> mappingMap = ImmutableMap.of(
            "properties",
            ImmutableMap.of(
                trainingField,
                ImmutableMap.of("type", KNNVectorFieldMapper.CONTENT_TYPE, KNNConstants.MODEL_ID, trainingFieldModeId)
            )
        );
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(mappingMap);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);
        DiscoveryNodes discoveryNodes = mock(DiscoveryNodes.class);
        when(discoveryNodes.getDataNodes()).thenReturn(Map.of());

        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        when(clusterState.nodes()).thenReturn(discoveryNodes);

        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        // Initialize static components with the mocks
        TrainingModelRequest.initialize(modelDao, clusterService);

        // Test that validation produces model already exists error message
        ActionRequestValidationException exception = trainingModelRequest.validate();
        assertNull(exception);
    }

    /**
     * This method produces a cluster service that will mock so that there are no validation exceptions.
     *
     * @param trainingIndex Name of training index
     * @param trainingField Name of training field
     * @param dimension Expected dimension
     * @return ClusterService
     */
    private ClusterService getClusterServiceForValidReturns(String trainingIndex, String trainingField, int dimension) {
        Map<String, Object> mappingMap = ImmutableMap.of(
            "properties",
            ImmutableMap.of(trainingField, ImmutableMap.of("type", KNNVectorFieldMapper.CONTENT_TYPE, KNNConstants.DIMENSION, dimension))
        );
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(mappingMetadata.getSourceAsMap()).thenReturn(mappingMap);
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(trainingIndex)).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.metadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        return clusterService;
    }
}
