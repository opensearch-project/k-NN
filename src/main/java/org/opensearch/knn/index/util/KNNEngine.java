/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.KNNMethod;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.training.VectorSpaceInfo;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;

/**
 * KNNEngine provides the functionality to validate and transform user defined indices into information that can be
 * passed to the respective k-NN library's JNI layer.
 */
public enum KNNEngine implements KNNLibrary {
    NMSLIB(NMSLIB_NAME, Nmslib.INSTANCE),
    FAISS(FAISS_NAME, Faiss.INSTANCE),
    LUCENE(LUCENE_NAME, Lucene.INSTANCE);

    public static final KNNEngine DEFAULT = NMSLIB;

    private static final Set<KNNEngine> CUSTOM_SEGMENT_FILE_ENGINES = ImmutableSet.of(KNNEngine.NMSLIB, KNNEngine.FAISS);
    private static final Set<KNNEngine> ENGINES_SUPPORTING_FILTERS = ImmutableSet.of(KNNEngine.LUCENE, KNNEngine.FAISS);
    public static final Set<KNNEngine> ENGINES_SUPPORTING_RADIAL_SEARCH = ImmutableSet.of(KNNEngine.LUCENE, KNNEngine.FAISS);

    private static Map<KNNEngine, Integer> MAX_DIMENSIONS_BY_ENGINE = Map.of(
        KNNEngine.NMSLIB,
        16_000,
        KNNEngine.FAISS,
        16_000,
        KNNEngine.LUCENE,
        16_000
    );

    /**
     * Constructor for KNNEngine
     *
     * @param name name of engine
     * @param knnLibrary library the engine uses
     */
    KNNEngine(String name, KNNLibrary knnLibrary) {
        this.name = name;
        this.knnLibrary = knnLibrary;
    }

    private final String name;
    private final KNNLibrary knnLibrary;

    /**
     * Get the engine
     *
     * @param name of engine to be fetched
     * @return KNNEngine corresponding to name
     */
    public static KNNEngine getEngine(String name) {
        if (NMSLIB.getName().equalsIgnoreCase(name)) {
            return NMSLIB;
        }

        if (FAISS.getName().equalsIgnoreCase(name)) {
            return FAISS;
        }

        if (LUCENE.getName().equalsIgnoreCase(name)) {
            return LUCENE;
        }

        throw new IllegalArgumentException(String.format("Invalid engine type: %s", name));
    }

    /**
     * Get the engine from the path.
     *
     * @param path to be checked
     * @return KNNEngine corresponding to path
     */
    public static KNNEngine getEngineNameFromPath(String path) {
        if (path.endsWith(KNNEngine.NMSLIB.getExtension()) || path.endsWith(KNNEngine.NMSLIB.getCompoundExtension())) {
            return KNNEngine.NMSLIB;
        }

        if (path.endsWith(KNNEngine.FAISS.getExtension()) || path.endsWith(KNNEngine.FAISS.getCompoundExtension())) {
            return KNNEngine.FAISS;
        }

        throw new IllegalArgumentException("No engine matches the path's suffix");
    }

    /**
     * Returns all engines that create custom segment files.
     *
     * @return Set of all engines that create custom segment files.
     */
    public static Set<KNNEngine> getEnginesThatCreateCustomSegmentFiles() {
        return CUSTOM_SEGMENT_FILE_ENGINES;
    }

    public static Set<KNNEngine> getEnginesThatSupportsFilters() {
        return ENGINES_SUPPORTING_FILTERS;
    }

    /**
     * Return number of max allowed dimensions per single vector based on the knn engine
     * @param knnEngine knn engine to check max dimensions value
     * @return
     */
    public static int getMaxDimensionByEngine(KNNEngine knnEngine) {
        return MAX_DIMENSIONS_BY_ENGINE.getOrDefault(knnEngine, MAX_DIMENSIONS_BY_ENGINE.get(KNNEngine.DEFAULT));
    }

    /**
     * Get the name of the engine
     *
     * @return name of the engine
     */
    public String getName() {
        return name;
    }

    @Override
    public String getVersion() {
        return knnLibrary.getVersion();
    }

    @Override
    public String getExtension() {
        return knnLibrary.getExtension();
    }

    @Override
    public String getCompoundExtension() {
        return knnLibrary.getCompoundExtension();
    }

    @Override
    public KNNMethod getMethod(String methodName) {
        return knnLibrary.getMethod(methodName);
    }

    @Override
    public EngineSpecificMethodContext getMethodContext(String methodName) {
        return knnLibrary.getMethodContext(methodName);
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return knnLibrary.score(rawScore, spaceType);
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        return knnLibrary.distanceToRadialThreshold(distance, spaceType);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return knnLibrary.scoreToRadialThreshold(score, spaceType);
    }

    @Override
    public ValidationException validateMethod(KNNMethodContext knnMethodContext) {
        return knnLibrary.validateMethod(knnMethodContext);
    }

    @Override
    public ValidationException validateMethodWithData(KNNMethodContext knnMethodContext, VectorSpaceInfo vectorSpaceInfo) {
        return knnLibrary.validateMethodWithData(knnMethodContext, vectorSpaceInfo);
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        return knnLibrary.isTrainingRequired(knnMethodContext);
    }

    @Override
    public Map<String, Object> getMethodAsMap(KNNMethodContext knnMethodContext) {
        return knnLibrary.getMethodAsMap(knnMethodContext);
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
        return knnLibrary.estimateOverheadInKB(knnMethodContext, dimension);
    }

    @Override
    public Boolean isInitialized() {
        return knnLibrary.isInitialized();
    }

    @Override
    public void setInitialized(Boolean isInitialized) {
        knnLibrary.setInitialized(isInitialized);
    }

    @Override
    public List<String> mmapFileExtensions() {
        return knnLibrary.mmapFileExtensions();
    }
}
