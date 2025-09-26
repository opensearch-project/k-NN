/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.mockito.ArgumentCaptor;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.transport.client.Client;

import java.util.*;

import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.getMMRFieldMappingByPath;

public class MMRUtilTests extends MMRTestCase {
    private Client mockClient;
    private ActionListener<MMRVectorFieldInfo> listener;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockClient = mock(Client.class);
        listener = mock(ActionListener.class);
    }

    public void testExtractVectorFromHit_whenValidList_thenReturnFloatArray() {
        Map<String, Object> source = new HashMap<>();
        source.put("embedding", Arrays.asList(0.1, 0.2, 0.3));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc1", true);

        assertArrayEquals(new float[] { 0.1f, 0.2f, 0.3f }, result, 0.0001f);
    }

    public void testExtractVectorFromHit_whenInvalidElementType_thenThrow() {
        Map<String, Object> source = new HashMap<>();
        source.put("embedding", Arrays.asList(1.0, "bad"));

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "embedding", "doc1", true)
        );
        assertTrue(ex.getMessage().contains("unexpected value at the vector field"));
    }

    public void testExtractVectorFromHit_whenFieldNotFound_thenThrow() {
        Map<String, Object> source = new HashMap<>();

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "missing", "doc1", true)
        );
        assertTrue(ex.getMessage().contains("not found"));
    }

    public void testResolveKnnVectorFieldInfo_whenAllUnmappedField_thenDefaultFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(Collections.emptyMap())),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.DEFAULT));
    }

    public void testResolveKnnVectorFieldInfo_whenAllUnmappedField_thenUserProvidedFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
        VectorDataType userProvidedVectorDataType = VectorDataType.FLOAT;

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(Collections.emptyMap())),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.COSINESIMIL, VectorDataType.FLOAT));
    }

    public void testResolveKnnVectorFieldInfo_whenNonKnnField_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of("properties", Map.of("field", Map.of(TYPE, "keyword")));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError = "MMR query extension cannot support non knn_vector field [index:field].";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentSpaceTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue()))
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    KNN_METHOD,
                    Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.COSINESIMIL.getValue())
                )
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different space type [l2, cosinesimil] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentVectorDataTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BINARY.getValue(),
                    TOP_LEVEL_PARAMETER_SPACE_TYPE,
                    SpaceType.L2.getValue()
                )
            )
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.FLOAT.getValue(),
                    KNN_METHOD,
                    Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                )
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different vector data type [binary, float] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentUserProvidedSpaceTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue()))
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The space type [cosinesimil] provided in the MMR query extension does not match the space type [l2] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentUserProvidedVectorDataTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = VectorDataType.FLOAT;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, VECTOR_DATA_TYPE_FIELD, VectorDataType.BYTE.getValue()))
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The vector data type [float] provided in the MMR query extension does not match the vector data type [byte] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenMappedFieldNoInfo_thenDefaultFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of("properties", Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE)));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.DEFAULT));
    }

    public void testResolveKnnVectorFieldInfo_whenMappedFieldWithModelId_thenFieldInfoFromModel() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        String modelId = "modelId";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId))
        );
        mockModelMetadata(mockClient, Map.of(modelId, new MMRVectorFieldInfo(SpaceType.HAMMING, VectorDataType.BINARY)));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.HAMMING, VectorDataType.BINARY));
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentModelSpaceTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        String modelId1 = "model1";
        String modelId2 = "model2";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId2))
        );
        mockModelMetadata(
            mockClient,
            Map.of(
                modelId1,
                new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT),
                modelId2,
                new MMRVectorFieldInfo(SpaceType.COSINESIMIL, VectorDataType.FLOAT)
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different space type [l2, cosinesimil] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentModelVectorDataTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        String modelId1 = "model1";
        String modelId2 = "model2";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId2))
        );
        mockModelMetadata(
            mockClient,
            Map.of(
                modelId1,
                new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT),
                modelId2,
                new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.BINARY)
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different vector data type [float, binary] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentSpaceTypeFromModelAndUser_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
        VectorDataType userProvidedVectorDataType = null;
        String modelId1 = "model1";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
        );
        mockModelMetadata(mockClient, Map.of(modelId1, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT)));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The space type [cosinesimil] provided in the MMR query extension does not match the space type [l2] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentVectorDataTypeFromModelAndUser_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = VectorDataType.BINARY;
        String modelId1 = "model1";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
        );
        mockModelMetadata(mockClient, Map.of(modelId1, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT)));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The vector data type [binary] provided in the MMR query extension does not match the vector data type [float] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenModelNotFount_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        String modelId1 = "model1";
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
        );
        mockModelMetadata(mockClient, Collections.emptyMap());

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "Failed to retrieve model(s) to resolve the space type and vector data type for the MMR query extension. Errors: Model ID model1 not found.";
        verifyException(listener, RuntimeException.class, expectedError);
    }

    private IndexMetadata createMockIndexMetadata(Map<String, Object> mappings) {
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(indexMetadata.getIndex()).thenReturn(new Index("index", "uuid"));
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        when(mappingMetadata.sourceAsMap()).thenReturn(mappings);
        return indexMetadata;
    }

    private void verifyVectorFieldInfo(ActionListener<MMRVectorFieldInfo> listener, MMRVectorFieldInfo vectorFieldInfo) {
        ArgumentCaptor<MMRVectorFieldInfo> captor = ArgumentCaptor.forClass(MMRVectorFieldInfo.class);
        verify(listener).onResponse(captor.capture());
        SpaceType capturedSpaceType = captor.getValue().getSpaceType();
        VectorDataType capturedVectorDataType = captor.getValue().getVectorDataType();
        assertEquals(vectorFieldInfo.getSpaceType(), capturedSpaceType);
        assertEquals(vectorFieldInfo.getVectorDataType(), capturedVectorDataType);
    }

    public void testShouldGenerateMMRProcessor_whenExtContainsBuilder_thenReturnTrue() {
        SearchRequest searchRequest = new SearchRequest();
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        searchSourceBuilder.ext(Collections.singletonList(new MMRSearchExtBuilder.Builder().build()));
        searchRequest.source(searchSourceBuilder);

        ProcessorGenerationContext ctx = new ProcessorGenerationContext(searchRequest);

        assertTrue(MMRUtil.shouldGenerateMMRProcessor(ctx));
    }

    public void testShouldGenerateMMRProcessor_whenNoExt_thenReturnFalse() {
        SearchRequest searchRequest = new SearchRequest();

        ProcessorGenerationContext ctx = new ProcessorGenerationContext(searchRequest);

        assertFalse(MMRUtil.shouldGenerateMMRProcessor(ctx));
    }

    public void testGetMMRFieldMappingByPath_whenInNestedField_thenException() {
        Map<String, Object> mappings = new HashMap<>();
        Map<String, Object> userMapping = new HashMap<>();
        userMapping.put("type", "nested");

        Map<String, Object> properties = new HashMap<>();
        properties.put("user", userMapping);
        mappings.put("properties", properties);

        String fieldPath = "user.profile.age";

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> getMMRFieldMappingByPath(mappings, fieldPath));

        String expectedError = "MMR search extension cannot support the field user.profile.age because it is in the nested field user.";
        assertEquals(expectedError, ex.getMessage());
    }
}
