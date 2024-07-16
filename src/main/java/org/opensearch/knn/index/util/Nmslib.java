/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.Parameter;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * Implements NativeLibrary for the nmslib native library
 */
class Nmslib extends NativeLibrary {

    // Extension to be used for Nmslib files. It is ".hnsw" and not ".nmslib" for legacy purposes.
    final static String EXTENSION = ".hnsw";

    final static String CURRENT_VERSION = "2011";

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(
        METHOD_HNSW,
        KNNMethod.Builder.builder(
            MethodComponent.Builder.builder(METHOD_HNSW)
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
        ).addSpaces(SpaceType.UNDEFINED, SpaceType.L2, SpaceType.L1, SpaceType.LINF, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT).build()
    );

    final static Nmslib INSTANCE = new Nmslib(METHODS, Collections.emptyMap(), CURRENT_VERSION, EXTENSION);

    /**
     * Constructor for Nmslib
     *
     * @param methods Set of methods the native library supports
     * @param scoreTranslation Map of translation of space type to scores returned by the library
     * @param currentVersion String representation of current version of the library
     * @param extension String representing the extension that library files should use
     */
    private Nmslib(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String currentVersion,
        String extension
    ) {
        super(methods, Map.of(METHOD_HNSW, new DefaultHnswContext()), scoreTranslation, currentVersion, extension);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }
}
