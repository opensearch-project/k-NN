/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.nmslib;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Implements NativeLibrary for the nmslib native library
 *
 * @deprecated As of 2.19.0, please use {@link org.opensearch.knn.index.engine.faiss.Faiss} or Lucene's native k-NN.
 * This engine will be removed in a future release.
 */
@Deprecated(since = "2.19.0", forRemoval = true)
public class Nmslib extends NativeLibrary {
    // Extension to be used for Nmslib files. It is ".hnsw" and not ".nmslib" for legacy purposes.
    public final static String EXTENSION = ".hnsw";
    final static String CURRENT_VERSION = "2011";

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HNSW, new NmslibHNSWMethod());

    public final static Nmslib INSTANCE = new Nmslib(METHODS, Collections.emptyMap(), CURRENT_VERSION, EXTENSION);
    private final MethodResolver methodResolver;

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
        super(methods, scoreTranslation, currentVersion, extension);
        this.methodResolver = new NmslibMethodResolver();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return distance;
    }

    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }
}
