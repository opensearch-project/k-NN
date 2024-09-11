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

package org.opensearch.knn.index.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.knn.index.KNNSettings;

/**
 * This class acts as an abstraction to get the default hyperparameter values for different parameters used in the
 * Nearest Neighbor Algorithm across different version of Index.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class IndexHyperParametersUtil {

    private static final int INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION_OLD_VALUE = 512;
    private static final int INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH_OLD_VALUE = 512;
    private static final int INDEX_BINARY_QUANTIZATION_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION = 256;
    private static final int INDEX_BINARY_QUANTIZATION_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH = 256;

    /**
     * Returns the default value of EF Construction that should be used for the input index version. After version 2.12.0
     * of Opensearch we are have reduced the value of ef_construction in favor of better build times.
     *
     * @param indexVersion {@code Version} of the index with which it was created.
     * @return default value of EF Construction that should be used for the input index version.
     */
    public static int getHNSWEFConstructionValue(@NonNull final Version indexVersion) {
        if (indexVersion.before(Version.V_2_12_0)) {
            log.debug(
                "Picking up old values of ef_construction : index version : {}, value: {}",
                indexVersion,
                INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION_OLD_VALUE
            );
            return INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION_OLD_VALUE;
        }
        log.debug(
            "Picking up new values of ef_construction : index version : {}, value: {}",
            indexVersion,
            KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION
        );
        return KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION;
    }

    /**
     * Returns the default value of EF Search that should be used for the input index version. After version 2.12.0
     * of Opensearch we are have reduced the value of ef_search in favor of better latency.
     *
     * @param indexVersion {@code Version} of the index with which it was created.
     * @return default value of EF Search that should be used for the input index version.
     */
    public static int getHNSWEFSearchValue(@NonNull final Version indexVersion) {
        if (indexVersion.before(Version.V_2_12_0)) {
            log.debug(
                "Picking up old values of ef_search : index version : {}, value: {}",
                indexVersion,
                INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH_OLD_VALUE
            );
            return INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH_OLD_VALUE;
        }
        log.debug(
            "Picking up new values of ef_search : index version : {}, value: {}",
            indexVersion,
            KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH
        );
        return KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH;
    }

    /*
     * Returns the default value of EF Construction that should be used with Binary Quantization.
     *
     * @return default value of EF Construction
     */
    public static int getBinaryQuantizationEFConstructionValue() {
        return INDEX_BINARY_QUANTIZATION_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION;
    }

    /*
     * Returns the default value of EF Search that should be used  with Binary Quantization.
     *
     * @return default value of EF Search
     */
    public static int getBinaryQuantizationEFSearchValue() {
        return INDEX_BINARY_QUANTIZATION_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH;
    }
}
