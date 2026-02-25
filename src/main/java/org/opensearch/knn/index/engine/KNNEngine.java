/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableSet;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

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
    @Deprecated(since = "2.19.0", forRemoval = true)
    NMSLIB(NMSLIB_NAME, Nmslib.INSTANCE, Version.V_3_0_0),
    FAISS(FAISS_NAME, Faiss.INSTANCE),
    LUCENE(LUCENE_NAME, Lucene.INSTANCE),
    UNDEFINED("undefined");

    public static final KNNEngine DEFAULT = FAISS;
    private final Version restrictedFromVersion; // Nullable field

    private static final Set<KNNEngine> CUSTOM_SEGMENT_FILE_ENGINES = ImmutableSet.of(KNNEngine.NMSLIB, KNNEngine.FAISS);
    private static final Set<KNNEngine> ENGINES_SUPPORTING_FILTERS = ImmutableSet.of(KNNEngine.LUCENE, KNNEngine.FAISS);
    public static final Set<KNNEngine> ENGINES_SUPPORTING_RADIAL_SEARCH = ImmutableSet.of(KNNEngine.LUCENE, KNNEngine.FAISS);
    public static final Set<KNNEngine> DEPRECATED_ENGINES = ImmutableSet.of(KNNEngine.NMSLIB);
    public static final Set<KNNEngine> ENGINES_SUPPORTING_NESTED_FIELDS = ImmutableSet.of(KNNEngine.LUCENE, KNNEngine.FAISS);

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
        this.restrictedFromVersion = null;
    }

    /**
     * Constructor for deprecated engines.
     */
    KNNEngine(String name, KNNLibrary knnLibrary, Version restrictedVersion) {
        this.name = name;
        this.knnLibrary = knnLibrary;
        this.restrictedFromVersion = restrictedVersion;
    }

    /**
     * Constructor for undefined engines.
     */
    KNNEngine(String name) {
        this.name = name;
        this.knnLibrary = null;
        this.restrictedFromVersion = null;
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

        if (UNDEFINED.getName().equalsIgnoreCase(name)) {
            return UNDEFINED;
        }

        throw new IllegalArgumentException(String.format("Invalid engine type: %s", name));
    }

    /**
     * Checks if the KNN engine is deprecated for a given OpenSearch version.
     *
     * @param indexVersionCreated The OpenSearch version in which the index is being created.
     * @return {@code true} if the engine is deprecated in the specified version or later, {@code false} otherwise.
     */
    @Override
    public boolean isRestricted(Version indexVersionCreated) {
        return restrictedFromVersion != null && indexVersionCreated.onOrAfter(restrictedFromVersion);
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

    /**
     * Get the Deprecated Version
     *
     * @return Deprecated Version
     */
    public Version getRestrictedFromVersion() {
        return restrictedFromVersion;
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
    public ValidationException validateMethod(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return knnLibrary.validateMethod(knnMethodContext, knnMethodConfigContext);
    }

    @Override
    public boolean isTrainingRequired(KNNMethodContext knnMethodContext) {
        return knnLibrary.isTrainingRequired(knnMethodContext);
    }

    @Override
    public KNNLibraryIndexingContext getKNNLibraryIndexingContext(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return knnLibrary.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @Override
    public KNNLibrarySearchContext getKNNLibrarySearchContext(String methodName) {
        return knnLibrary.getKNNLibrarySearchContext(methodName);
    }

    @Override
    public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        return knnLibrary.estimateOverheadInKB(knnMethodContext, knnMethodConfigContext);
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

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        return knnLibrary.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }

    @Override
    public boolean supportsRemoteIndexBuild(KNNLibraryIndexingContext knnLibraryIndexingContext) {
        return knnLibrary.supportsRemoteIndexBuild(knnLibraryIndexingContext);
    }

    @Override
    public RemoteIndexParameters createRemoteIndexingParameters(Map<String, Object> parameters) {
        return knnLibrary.createRemoteIndexingParameters(parameters);
    }

    @Override
    public VectorSearcherFactory getVectorSearcherFactory() {
        return knnLibrary.getVectorSearcherFactory();
    }
}
