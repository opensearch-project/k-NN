/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.common.Explicit;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.util.KNNEngine;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_M;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Field mapper for original implementation
 */
public class LegacyFieldMapper extends KNNVectorFieldMapper {

    protected String spaceType;
    protected String m;
    protected String efConstruction;

    LegacyFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        String spaceType,
        String m,
        String efConstruction
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo, ignoreMalformed, stored, hasDocValues);

        this.spaceType = spaceType;
        this.m = m;
        this.efConstruction = efConstruction;

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);

        this.fieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        this.fieldType.putAttribute(SPACE_TYPE, spaceType);
        this.fieldType.putAttribute(KNN_ENGINE, KNNEngine.NMSLIB.getName());

        // These are extra just for legacy
        this.fieldType.putAttribute(HNSW_ALGO_M, m);
        this.fieldType.putAttribute(HNSW_ALGO_EF_CONSTRUCTION, efConstruction);

        this.fieldType.freeze();
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new KNNVectorFieldMapper.Builder(simpleName(), this.spaceType, this.m, this.efConstruction).init(this);
    }

    static String getSpaceType(Settings indexSettings) {
        String spaceType = indexSettings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey());
        if (spaceType == null) {
            logger.info(
                "[KNN] The setting \""
                    + METHOD_PARAMETER_SPACE_TYPE
                    + "\" was not set for the index. "
                    + "Likely caused by recent version upgrade. Setting the setting to the default value="
                    + KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE
            );
            return KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE;
        }
        return spaceType;
    }

    static String getM(Settings indexSettings) {
        String m = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_M_SETTING.getKey());
        if (m == null) {
            logger.info(
                "[KNN] The setting \""
                    + HNSW_ALGO_M
                    + "\" was not set for the index. "
                    + "Likely caused by recent version upgrade. Setting the setting to the default value="
                    + KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M
            );
            return String.valueOf(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M);
        }
        return m;
    }

    static String getEfConstruction(Settings indexSettings) {
        String efConstruction = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING.getKey());
        if (efConstruction == null) {
            logger.info(
                "[KNN] The setting \""
                    + HNSW_ALGO_EF_CONSTRUCTION
                    + "\" was not set for"
                    + " the index. Likely caused by recent version upgrade. Setting the setting to the default value="
                    + KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION
            );
            return String.valueOf(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION);
        }
        return efConstruction;
    }
}
