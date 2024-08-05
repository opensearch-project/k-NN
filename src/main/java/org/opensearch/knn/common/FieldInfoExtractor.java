/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.indices.ModelUtil.getModelMetadata;

/**
 * A utility class to extract information from FieldInfo.
 */
@UtilityClass
public class FieldInfoExtractor {

    /**
     * Extracts KNNEngine from FieldInfo
     * @param field {@link FieldInfo}
     * @return {@link KNNEngine}
     */
    public static KNNEngine extractKNNEngine(final FieldInfo field) {
        final ModelMetadata modelMetadata = getModelMetadata(field.attributes().get(MODEL_ID));
        if (modelMetadata != null) {
            return modelMetadata.getKnnEngine();
        }
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }

    /**
     * Extracts VectorDataType from FieldInfo
     * @param fieldInfo {@link FieldInfo}
     * @return {@link VectorDataType}
     */
    public static VectorDataType extractVectorDataType(final FieldInfo fieldInfo) {
        String vectorDataTypeString = fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD);
        if (StringUtils.isEmpty(vectorDataTypeString)) {
            final ModelMetadata modelMetadata = ModelUtil.getModelMetadata(fieldInfo.getAttribute(KNNConstants.MODEL_ID));
            if (modelMetadata != null) {
                VectorDataType vectorDataType = modelMetadata.getVectorDataType();
                vectorDataTypeString = vectorDataType == null ? null : vectorDataType.getValue();
            }
        }
        return StringUtils.isNotEmpty(vectorDataTypeString) ? VectorDataType.get(vectorDataTypeString) : VectorDataType.DEFAULT;
    }
}
