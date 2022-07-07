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

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.*;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;

/**
 * Implements NativeLibrary for the nmslib native library
 */
class Nmslib extends NativeLibrary {
    // ======================================
    // Constants pertaining to nmslib library
    // ======================================
    public final static String HNSW_LIB_NAME = "hnsw";
    public final static String EXTENSION = ".hnsw";

    public final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
        METHOD_HNSW,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(HNSW_LIB_NAME)
                .addParameter(
                    METHOD_PARAMETER_M,
                    new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0)
                )
                .addParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    new Parameter.IntegerParameter(
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                        v -> v > 0
                    )
                )
                .build()
        ).addSpaces(SpaceType.L2, SpaceType.L1, SpaceType.LINF, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT).build()
    );

    public final static Nmslib INSTANCE = new Nmslib(
        METHODS,
        Collections.emptyMap(),
        Version.LATEST.getBuildVersion(),
        Version.LATEST.indexLibraryVersion(),
        EXTENSION
    );

    /**
     * Constructor for Nmslib
     *
     * @param methods Set of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param latestLibraryBuildVersion String representation of latest build version of the library
     * @param latestLibraryVersion String representation of latest version of the library
     * @param extension String representing the extension that library files should use
     */
    private Nmslib(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String latestLibraryBuildVersion,
        String latestLibraryVersion,
        String extension
    ) {
        super(methods, scoreTranslation, latestLibraryBuildVersion, latestLibraryVersion, extension);
    }

    public enum Version implements LibVersion {
        V2011("2011");

        public static final Version LATEST = V2011;

        private final String buildVersion;

        Version(String buildVersion) {
            this.buildVersion = buildVersion;
        }

        public String indexLibraryVersion() {
            return KNNConstants.NMSLIB_JNI_LIBRARY_NAME;
        }

        public String getBuildVersion() {
            return buildVersion;
        }
    }
}
