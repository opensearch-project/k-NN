/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.Explicit;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.KNNClusterUtil;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.knn.index.util.KNNEngine;

import java.security.InvalidParameterException;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.IndexScope;
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
    // Settings related to this field mapping type
    public static final String KNN_SPACE_TYPE = "index.knn.space_type";
    public static final String INDEX_KNN_DEFAULT_SPACE_TYPE = "l2";
    public static final Setting<String> INDEX_KNN_SPACE_TYPE = Setting.simpleString(KNN_SPACE_TYPE, INDEX_KNN_DEFAULT_SPACE_TYPE, s -> {
        try {
            SpaceType.getSpace(s);
        } catch (IllegalArgumentException ex) {
            throw new InvalidParameterException(ex.getMessage());
        }
    }, IndexScope, Setting.Property.Deprecated);

    public static final String KNN_ALGO_PARAM_M = "index.knn.algo_param.m";
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_M = 16;
    /**
     * M - the number of bi-directional links created for every new element during construction.
     * Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic
     * dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls.
     * The parameter also determines the algorithm's memory consumption, which is roughly M * 8-10 bytes per stored element.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_M_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_M,
        INDEX_KNN_DEFAULT_ALGO_PARAM_M,
        2,
        IndexScope,
        Setting.Property.Deprecated
    );

    public static final String KNN_ALGO_PARAM_EF_CONSTRUCTION = "index.knn.algo_param.ef_construction";
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION = 100;
    /**
     * ef_constrution - the parameter has the same meaning as ef, but controls the index_time/index_accuracy.
     * Bigger ef_construction leads to longer construction(more indexing time), but better index quality.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_EF_CONSTRUCTION,
        INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
        2,
        IndexScope,
        Setting.Property.Deprecated
    );

    public static final String KNN_ALGO_PARAM_EF_SEARCH = "index.knn.algo_param.ef_search";
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH = 100;
    /**
     * ef or efSearch - the size of the dynamic list for the nearest neighbors (used during the search).
     * Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k.
     * The value ef can be anything between k and the size of the dataset.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_EF_SEARCH,
        INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
        2,
        IndexScope,
        Dynamic
    );

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

    static String getSpaceType(Settings indexSettings) {
        String spaceType = indexSettings.get(INDEX_KNN_SPACE_TYPE.getKey());
        if (spaceType == null) {
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    METHOD_PARAMETER_SPACE_TYPE,
                    INDEX_KNN_DEFAULT_SPACE_TYPE
                )
            );
            return INDEX_KNN_DEFAULT_SPACE_TYPE;
        }
        return spaceType;
    }

    static String getM(Settings indexSettings) {
        String m = indexSettings.get(INDEX_KNN_ALGO_PARAM_M_SETTING.getKey());
        if (m == null) {
            log.info(
                String.format(
                    "[KNN] The setting \"%s\" was not set for the index. Likely caused by recent version upgrade. Setting the setting to the default value=%s",
                    HNSW_ALGO_M,
                    INDEX_KNN_DEFAULT_ALGO_PARAM_M
                )
            );
            return String.valueOf(INDEX_KNN_DEFAULT_ALGO_PARAM_M);
        }
        return m;
    }

    static String getEfConstruction(Settings indexSettings, Version indexVersion) {
        final String efConstruction = indexSettings.get(INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING.getKey());
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

    /**
     * @param index Name of the index
     * @return efSearch value
     */
    public static int getEfSearchParam(String index) {
        final IndexMetadata indexMetadata = KNNClusterUtil.instance().getIndexMetadata(index);
        return indexMetadata.getSettings()
            .getAsInt(KNN_ALGO_PARAM_EF_SEARCH, IndexHyperParametersUtil.getHNSWEFSearchValue(indexMetadata.getCreationVersion()));
    }
}
