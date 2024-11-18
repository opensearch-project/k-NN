/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.indices.ModelUtil.getModelMetadata;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import org.opensearch.knn.indices.ModelDao;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

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

    /**
     * Extract quantization config from fieldInfo
     *
     * @param fieldInfo {@link FieldInfo}
     * @return {@link QuantizationConfig}
     */
    public static QuantizationConfig extractQuantizationConfig(final FieldInfo fieldInfo) {
        String quantizationConfigString = fieldInfo.getAttribute(QFRAMEWORK_CONFIG);
        if (StringUtils.isEmpty(quantizationConfigString)) {
            return QuantizationConfig.EMPTY;
        }
        return QuantizationConfigParser.fromCsv(quantizationConfigString);
    }

    /**
     * Get the space type for the given field info.
     *
     * @param modelDao ModelDao instance to retrieve model metadata
     * @param fieldInfo FieldInfo instance to extract space type from
     * @return SpaceType for the given field info
     */
    public static SpaceType getSpaceType(final ModelDao modelDao, final FieldInfo fieldInfo) {
        final String spaceTypeString = fieldInfo.getAttribute(SPACE_TYPE);
        if (StringUtils.isNotEmpty(spaceTypeString)) {
            return SpaceType.getSpace(spaceTypeString);
        }

        final String modelId = fieldInfo.getAttribute(MODEL_ID);
        if (StringUtils.isEmpty(modelId)) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Unable to find the Space Type from Field Info attribute for field %s", fieldInfo.getName())
            );
        }

        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (modelMetadata == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unable to find the model metadata for model id %s", modelId));
        }
        return modelMetadata.getSpaceType();
    }

    public static String getParameters(final FieldInfo fieldInfo) {
        final String parameters = fieldInfo.getAttribute(PARAMETERS);
        return parameters;
    }
}
