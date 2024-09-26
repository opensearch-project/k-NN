/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;

/**
 * Implements NativeLibrary for the faiss native library
 */
public class Faiss extends NativeLibrary {
    public static final String FAISS_BINARY_INDEX_DESCRIPTION_PREFIX = "B";
    Map<SpaceType, Function<Float, Float>> scoreTransform;

    // TODO: Current version is not really current version. Instead, it encodes information in the file name
    // about the compatibility version the file is created with. In the future, we should refactor this so that it
    // makes sense. See https://github.com/opensearch-project/k-NN/issues/1515 for more details.
    private final static String CURRENT_VERSION = "165";

    // Map that overrides OpenSearch score translation by space type of scores returned by faiss
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = ImmutableMap.of(
        SpaceType.INNER_PRODUCT,
        rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1 * rawScore)
    );

    // Map that overrides radial search score threshold to faiss required distance, check more details in knn documentation:
    // https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/#spaces
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TO_DISTANCE_TRANSFORMATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder().put(SpaceType.INNER_PRODUCT, score -> score > 1 ? 1 - score : 1 / score - 1).build();

    // Package private so that the method resolving logic can access the methods
    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HNSW, new FaissHNSWMethod(), METHOD_IVF, new FaissIVFMethod());

    public final static Faiss INSTANCE = new Faiss(
        METHODS,
        SCORE_TRANSLATIONS,
        CURRENT_VERSION,
        KNNConstants.FAISS_EXTENSION,
        SCORE_TO_DISTANCE_TRANSFORMATIONS
    );

    private final MethodResolver methodResolver;

    /**
     * Constructor for Faiss
     *
     * @param methods                   map of methods the native library supports
     * @param scoreTranslation          Map of translation of space type to scores returned by the library
     * @param currentVersion            String representation of current version of the library
     * @param extension                 String representing the extension that library files should use
     */
    private Faiss(
        Map<String, KNNMethod> methods,
        Map<SpaceType, Function<Float, Float>> scoreTranslation,
        String currentVersion,
        String extension,
        Map<SpaceType, Function<Float, Float>> scoreTransform
    ) {
        super(methods, scoreTranslation, currentVersion, extension);
        this.scoreTransform = scoreTransform;
        this.methodResolver = new FaissMethodResolver();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        // Faiss engine uses distance as is and does not need transformation
        return distance;
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        // Faiss engine uses distance as is and need transformation
        if (this.scoreTransform.containsKey(spaceType)) {
            return this.scoreTransform.get(spaceType).apply(score);
        }
        return spaceType.scoreToDistanceTranslation(score);
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
