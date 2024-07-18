/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.knn.index.util.KNNEngine;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_M;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Field mapper for original implementation. It defaults to using nmslib as the engine and retrieves parameters from index settings.
 *
 * Example of this mapper output:
 *
 *   {
 *    "type": "knn_vector",
 *    "dimension": 128
 *   }
 */
@Log4j2
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
        String efConstruction,
        Version indexCreatedVersion
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo, ignoreMalformed, stored, hasDocValues, indexCreatedVersion);

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
        return new KNNVectorFieldMapper.Builder(simpleName(), this.spaceType, this.m, this.efConstruction, this.indexCreatedVersion).init(
            this
        );
    }

    static String getSpaceType(final Settings indexSettings, final VectorDataType vectorDataType) {
        String spaceType = indexSettings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey());
        if (spaceType == null) {
            spaceType = VectorDataType.BINARY == vectorDataType
                ? KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE_FOR_BINARY
                : KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE;
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    METHOD_PARAMETER_SPACE_TYPE,
                    spaceType
                )
            );
        }
        return spaceType;
    }

    static String getM(Settings indexSettings) {
        String m = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_M_SETTING.getKey());
        if (m == null) {
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    HNSW_ALGO_M,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M
                )
            );
            return String.valueOf(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M);
        }
        return m;
    }

    static String getEfConstruction(Settings indexSettings, Version indexVersion) {
        final String efConstruction = indexSettings.get(KNNSettings.INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING.getKey());
        if (efConstruction == null) {
            final String defaultEFConstructionValue = String.valueOf(IndexHyperParametersUtil.getHNSWEFConstructionValue(indexVersion));
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. "
                        + "Picking up default value for the index =%s",
                    HNSW_ALGO_EF_CONSTRUCTION,
                    defaultEFConstructionValue
                )
            );
            return defaultEFConstructionValue;
        }
        return efConstruction;
    }
}
