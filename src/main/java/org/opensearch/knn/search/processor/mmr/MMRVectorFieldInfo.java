/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.engine.SpaceTypeResolver.getDefaultSpaceType;

/**
 * A DTO to hold the info of the vector field used for MMR
 */
@Data
@NoArgsConstructor
public class MMRVectorFieldInfo {
    private String indexName;
    private String fieldPath;
    private VectorDataType vectorDataType;
    private String modelId;
    private SpaceType spaceType;
    private boolean unmapped;
    private String fieldType;

    public MMRVectorFieldInfo(SpaceType spaceType, VectorDataType vectorDataType) {
        this.spaceType = spaceType;
        this.vectorDataType = vectorDataType;
    }

    public boolean isKNNVectorField() {
        return KNNVectorFieldMapper.CONTENT_TYPE.equals(fieldType);
    }

    public void setKnnConfig(@NonNull final Map<String, Object> knnConfig) {
        setVectorDataTypeByConfig(knnConfig);
        if (setModelIdIfPresent(knnConfig)) {
            return;
        }
        if (setSpaceTypeIfPresent(knnConfig)) {
            return;
        }
        this.spaceType = getDefaultSpaceType(vectorDataType);
    }

    public void setIndexNameByIndexMetadata(@NonNull final IndexMetadata indexMetadata) {
        this.indexName = indexMetadata.getIndex().getName();
    }

    private void setVectorDataTypeByConfig(Map<String, Object> knnConfig) {
        String dataType = (String) knnConfig.get(VECTOR_DATA_TYPE_FIELD);
        this.vectorDataType = (dataType == null) ? VectorDataType.DEFAULT : VectorDataType.get(dataType);
    }

    private boolean setModelIdIfPresent(Map<String, Object> knnConfig) {
        String modelId = (String) knnConfig.get(MODEL_ID);
        if (modelId != null) {
            this.modelId = modelId;
            return true;
        }
        return false;
    }

    private boolean setSpaceTypeIfPresent(Map<String, Object> knnConfig) {
        String topLevelSpaceType = (String) knnConfig.get(TOP_LEVEL_PARAMETER_SPACE_TYPE);
        if (topLevelSpaceType != null) {
            this.spaceType = SpaceType.getSpace(topLevelSpaceType);
            return true;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> knnMethod = (Map<String, Object>) knnConfig.get(KNN_METHOD);
        if (knnMethod != null) {
            String spaceType = (String) knnMethod.get(METHOD_PARAMETER_SPACE_TYPE);
            if (spaceType != null) {
                this.spaceType = SpaceType.getSpace(spaceType);
                return true;
            }
        }
        return false;
    }

}
