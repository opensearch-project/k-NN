/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.Version;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.indices.ModelUtil.getModelMetadata;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import org.opensearch.knn.indices.ModelDao;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * A utility class to extract information from FieldInfo and also provides utility functions to extract fieldInfo
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
            } else if (fieldInfo.hasVectorValues()) {
                vectorDataTypeString = fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32
                    ? VectorDataType.FLOAT.toString()
                    : VectorDataType.BYTE.toString();
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
    public static QuantizationConfig extractQuantizationConfig(final FieldInfo fieldInfo, Version luceneVersion) {
        String quantizationConfigString = fieldInfo.getAttribute(QFRAMEWORK_CONFIG);
        if (StringUtils.isEmpty(quantizationConfigString)) {
            return QuantizationConfig.EMPTY;
        }
        return QuantizationConfigParser.fromCsv(quantizationConfigString, luceneVersion);
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
        if (StringUtils.isNotEmpty(modelId)) {
            return getSpaceTypeFromModel(modelDao, modelId);
        }
        if (fieldInfo.getVectorSimilarityFunction() != null) {
            return SpaceType.getSpace(fieldInfo.getVectorSimilarityFunction());
        }
        throw new IllegalArgumentException(
            String.format(Locale.ROOT, "Unable to find the Space Type from Field Info attribute for field %s", fieldInfo.getName())
        );

    }

    private static SpaceType getSpaceTypeFromModel(final ModelDao modelDao, final String modelId) {
        final ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (modelMetadata == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unable to find the model metadata for model id %s", modelId));
        }
        return modelMetadata.getSpaceType();
    }

    /**
     * Get the field info for the given field name, do a null check on the fieldInfo, as this function can return null,
     * if the field is not found.
     * @param leafReader {@link LeafReader}
     * @param fieldName {@link String}
     * @return {@link FieldInfo}
     */
    public static @Nullable FieldInfo getFieldInfo(final LeafReader leafReader, final String fieldName) {
        return leafReader.getFieldInfos().fieldInfo(fieldName);
    }
}
