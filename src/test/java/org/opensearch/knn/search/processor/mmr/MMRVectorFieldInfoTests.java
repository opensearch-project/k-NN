/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.core.index.Index;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MMRVectorFieldInfoTests extends KNNTestCase {

    public void testConstructorWithSpaceTypeAndVectorDataType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT);

        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
    }

    public void testIsKNNVectorField_whenContentTypeMatches_thenTrue() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        info.setFieldType(KNNVectorFieldMapper.CONTENT_TYPE);

        assertTrue(info.isKNNVectorField());
    }

    public void testIsKNNVectorField_whenContentTypeDoesNotMatch_thenFalse() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        info.setFieldType("text");

        assertFalse(info.isKNNVectorField());
    }

    public void testSetKnnConfig_withModelId_thenSetsModelIdAndSkipsSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = new HashMap<>();
        knnConfig.put(MODEL_ID, "test-model-id");
        knnConfig.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue());

        info.setKnnConfig(knnConfig);

        assertEquals("test-model-id", info.getModelId());
        assertNull(info.getSpaceType());
    }

    public void testSetKnnConfig_withTopLevelSpaceType_thenSetsSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = Map.of(TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.COSINESIMIL.getValue());

        info.setKnnConfig(knnConfig);

        assertEquals(SpaceType.COSINESIMIL, info.getSpaceType());
        assertNull(info.getModelId());
    }

    public void testSetKnnConfig_withMethodSpaceType_thenSetsSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnMethod = Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue());
        Map<String, Object> knnConfig = Map.of(KNN_METHOD, knnMethod);

        info.setKnnConfig(knnConfig);

        assertEquals(SpaceType.INNER_PRODUCT, info.getSpaceType());
    }

    public void testSetKnnConfig_withoutSpaceType_thenUsesDefaultSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = Map.of(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());

        info.setKnnConfig(knnConfig);

        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
        assertEquals(SpaceType.L2, info.getSpaceType());
    }

    public void testSetKnnConfig_withoutDataType_thenUsesDefaultVectorDataType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = Map.of(TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue());

        info.setKnnConfig(knnConfig);

        assertEquals(VectorDataType.DEFAULT, info.getVectorDataType());
    }

    public void testSetKnnConfig_withBinaryDataType_thenSetsBinaryType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = Map.of(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue());

        info.setKnnConfig(knnConfig);

        assertEquals(VectorDataType.BINARY, info.getVectorDataType());
        assertEquals(SpaceType.HAMMING, info.getSpaceType());
    }

    public void testIsKNNVectorField_whenFieldTypeNull_thenFalse() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();

        assertFalse(info.isKNNVectorField());
    }

    public void testSetKnnConfig_withEmptyConfig_thenUsesDefaults() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();

        info.setKnnConfig(Map.of());

        assertEquals(VectorDataType.DEFAULT, info.getVectorDataType());
        assertEquals(SpaceType.L2, info.getSpaceType());
        assertNull(info.getModelId());
    }

    public void testSetKnnConfig_withKnnMethodWithoutSpaceType_thenUsesDefaultSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnConfig = Map.of(KNN_METHOD, Map.of("name", "hnsw"));

        info.setKnnConfig(knnConfig);

        assertEquals(VectorDataType.DEFAULT, info.getVectorDataType());
        assertEquals(SpaceType.L2, info.getSpaceType());
    }

    public void testSetKnnConfig_withTopLevelSpaceTypePreferredOverMethodSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> knnMethod = Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue());
        Map<String, Object> knnConfig = new HashMap<>();
        knnConfig.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.COSINESIMIL.getValue());
        knnConfig.put(KNN_METHOD, knnMethod);

        info.setKnnConfig(knnConfig);

        assertEquals(SpaceType.COSINESIMIL, info.getSpaceType());
    }

    public void testSetIndexNameByIndexMetadata() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.getIndex()).thenReturn(new Index("test-index", "test-uuid"));

        info.setIndexNameByIndexMetadata(indexMetadata);

        assertEquals("test-index", info.getIndexName());
    }
}
