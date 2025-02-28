/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.NativeLibrary;
import org.opensearch.knn.index.engine.ResolvedMethodContext;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.ALGORITHM;
import static org.opensearch.knn.common.KNNConstants.ALGORITHM_PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE;

/**
 * Implements NativeLibrary for the faiss native library
 */
public class Faiss extends NativeLibrary {
    public static final String FAISS_BINARY_INDEX_DESCRIPTION_PREFIX = "B";
    Map<SpaceType, Function<Float, Float>> distanceTransform;
    Map<SpaceType, Function<Float, Float>> scoreTransform;

    // TODO: Current version is not really current version. Instead, it encodes information in the file name
    // about the compatibility version the file is created with. In the future, we should refactor this so that it
    // makes sense. See https://github.com/opensearch-project/k-NN/issues/1515 for more details.
    private final static String CURRENT_VERSION = "165";

    // Map that overrides OpenSearch score translation by space type of scores returned by faiss
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TRANSLATIONS = ImmutableMap.of(
        SpaceType.INNER_PRODUCT,
        rawScore -> SpaceType.INNER_PRODUCT.scoreTranslation(-1 * rawScore),
        // COSINESIMIL expects the raw score in 1 - cosine(x,y)
        SpaceType.COSINESIMIL,
        rawScore -> SpaceType.COSINESIMIL.scoreTranslation(1 - rawScore)
    );

    // Map that overrides radial search score threshold to faiss required distance, check more details in knn documentation:
    // https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/#spaces
    private final static Map<SpaceType, Function<Float, Float>> SCORE_TO_DISTANCE_TRANSFORMATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder()
        .put(SpaceType.INNER_PRODUCT, score -> score > 1 ? 1 - score : (1 / score) - 1)
        .put(SpaceType.COSINESIMIL, score -> 2 * score - 1)
        .build();

    private final static Map<SpaceType, Function<Float, Float>> DISTANCE_TRANSLATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder().put(SpaceType.COSINESIMIL, distance -> 1 - distance).build();

    // Package private so that the method resolving logic can access the methods
    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HNSW, new FaissHNSWMethod(), METHOD_IVF, new FaissIVFMethod());

    public final static Faiss INSTANCE = new Faiss(
        METHODS,
        SCORE_TRANSLATIONS,
        CURRENT_VERSION,
        KNNConstants.FAISS_EXTENSION,
        SCORE_TO_DISTANCE_TRANSFORMATIONS,
        DISTANCE_TRANSLATIONS
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
        Map<SpaceType, Function<Float, Float>> scoreTransform,
        Map<SpaceType, Function<Float, Float>> distanceTransform
    ) {
        super(methods, scoreTranslation, currentVersion, extension);
        this.scoreTransform = scoreTransform;
        this.distanceTransform = distanceTransform;
        this.methodResolver = new FaissMethodResolver();
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        if (this.distanceTransform.containsKey(spaceType)) {
            return this.distanceTransform.get(spaceType).apply(distance);
        }
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

    /**
     * Get the parameters that need to be passed to the remote build service for training
     * @param indexInfo to parse
     * @return Map of parameters to be used as "index_parameters"
     */
    @Override
    public Map<String, Object> getRemoteIndexingParameters(BuildIndexParams indexInfo) {
        Map<String, Object> indexParameters = new HashMap<>();
        String methodName = (String) indexInfo.getParameters().get(NAME);
        indexParameters.put(ALGORITHM, methodName);
        indexParameters.put(METHOD_PARAMETER_SPACE_TYPE, indexInfo.getParameters().getOrDefault(SPACE_TYPE, INDEX_KNN_DEFAULT_SPACE_TYPE));

        assert (indexInfo.getParameters().containsKey(PARAMETERS));
        Object innerParams = indexInfo.getParameters().get(PARAMETERS);
        assert (innerParams instanceof Map);
        {
            Map<String, Object> algorithmParams = new HashMap<>();
            Map<String, Object> innerMap = (Map<String, Object>) innerParams;
            switch (methodName) {
                case METHOD_HNSW -> {
                    algorithmParams.put(
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        innerMap.getOrDefault(METHOD_PARAMETER_EF_CONSTRUCTION, INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION)
                    );
                    algorithmParams.put(
                        METHOD_PARAMETER_EF_SEARCH,
                        innerMap.getOrDefault(METHOD_PARAMETER_EF_SEARCH, INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH)
                    );
                    Object indexDescription = indexInfo.getParameters().get(INDEX_DESCRIPTION_PARAMETER);
                    assert indexDescription instanceof String;
                    algorithmParams.put(METHOD_PARAMETER_M, getMFromIndexDescription((String) indexDescription));
                }
                case METHOD_IVF -> {
                    algorithmParams.put(
                        METHOD_PARAMETER_NLIST,
                        innerMap.getOrDefault(METHOD_PARAMETER_NLIST, METHOD_PARAMETER_NLIST_DEFAULT)
                    );
                    algorithmParams.put(
                        METHOD_PARAMETER_NPROBES,
                        innerMap.getOrDefault(METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NPROBES_DEFAULT)
                    );
                }
            }
            indexParameters.put(ALGORITHM_PARAMETERS, algorithmParams);
        }
        return indexParameters;
    }

    public static int getMFromIndexDescription(String indexDescription) {
        int commaIndex = indexDescription.indexOf(",");
        if (commaIndex == -1) {
            throw new IllegalArgumentException("Invalid index description: " + indexDescription);
        }
        String hnswPart = indexDescription.substring(0, commaIndex);
        int m = Integer.parseInt(hnswPart.substring(4));
        assert (m > 1 && m < 100);
        return m;
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

    @Override
    public boolean supportsRemoteIndexBuild() {
        return true;
    }
}
