/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;

/**
 * KNNEngine provides the functionality to validate and transform user defined indices into information that can be
 * passed to the respective k-NN library's JNI layer.
 *
 * <p>Historically this was a Java {@code enum} with a fixed set of constants. It is now an open, instance-based
 * type: the built-in engines are {@code public static final} singletons (so existing {@code KNNEngine.FAISS}
 * references and {@code ==} identity checks are unchanged), and additional engines can be contributed at
 * runtime through {@link KNNEngineDefinition} (discovered by {@link KNNEngineRegistry}). The public surface
 * ({@link #values()}, {@link #getName()}, {@link #getEngine(String)}, the capability accessors) is preserved.
 */
public final class KNNEngine implements KNNLibrary, VectorSearchEngine {

    // Built-ins pass a null native service: their JNI lifecycle is the core FaissService/NmslibService.
    // A runtime-registered engine carries its own NativeEngineService from its KNNEngineDefinition.
    @Deprecated(since = "2.19.0", forRemoval = true)
    public static final KNNEngine NMSLIB = new KNNEngine("NMSLIB", NMSLIB_NAME, Nmslib.INSTANCE, Version.V_3_0_0, null);
    public static final KNNEngine FAISS = new KNNEngine("FAISS", FAISS_NAME, Faiss.INSTANCE, null, null);
    public static final KNNEngine LUCENE = new KNNEngine("LUCENE", LUCENE_NAME, Lucene.INSTANCE, null, null);
    public static final KNNEngine UNDEFINED = new KNNEngine("UNDEFINED", "undefined", null, null, null);

    public static final KNNEngine DEFAULT = FAISS;

    // Built-in engines in declaration order. Registered engines (from KNNEngineRegistry) are appended after.
    private static final List<KNNEngine> BUILT_INS = List.of(NMSLIB, FAISS, LUCENE, UNDEFINED);

    // All engines (built-ins + any runtime-registered engine), resolved once at class load.
    private static final Map<String, KNNEngine> BY_NAME = buildRegistry();
    private static final KNNEngine[] ALL = BY_NAME.values().toArray(new KNNEngine[0]);

    private static Map<String, KNNEngine> buildRegistry() {
        final Map<String, KNNEngine> byName = new LinkedHashMap<>();
        for (KNNEngine engine : BUILT_INS) {
            final String key = engine.name.toLowerCase(Locale.ROOT);
            // KNNEngineRegistry keeps its own copy of the built-in names (it loads during this class's
            // initialization, so it cannot read BUILT_INS); this assert catches drift between the two.
            assert KNNEngineRegistry.BUILT_IN_ENGINE_NAMES.contains(key) : "Built-in engine ["
                + key
                + "] missing from KNNEngineRegistry.BUILT_IN_ENGINE_NAMES";
            byName.put(key, engine);
        }
        // Registered engines are fully materialized (and collision/failure-filtered) by KNNEngineRegistry.
        for (KNNEngineRegistry.RegisteredEngine registered : KNNEngineRegistry.all()) {
            byName.put(
                registered.engineName().toLowerCase(Locale.ROOT),
                new KNNEngine(
                    registered.engineName().toUpperCase(Locale.ROOT),
                    registered.engineName(),
                    registered.library(),
                    null,
                    registered.nativeService()
                )
            );
        }
        return java.util.Collections.unmodifiableMap(byName);
    }

    // ----- Capability sets, derived from each engine's KNNLibrary flags (never from engine identity) -----

    private static final Set<KNNEngine> CUSTOM_SEGMENT_FILE_ENGINES = enginesWith(KNNLibrary::createsCustomSegmentFiles);
    private static final Set<KNNEngine> ENGINES_SUPPORTING_FILTERS = enginesWith(KNNLibrary::supportsFilters);
    public static final Set<KNNEngine> ENGINES_SUPPORTING_RADIAL_SEARCH = enginesWith(KNNLibrary::supportsRadialSearch);
    public static final Set<KNNEngine> ENGINES_SUPPORTING_NESTED_FIELDS = enginesWith(KNNLibrary::supportsNestedFields);
    // Deprecation is core release policy, not an engine capability, so this set stays literal.
    public static final Set<KNNEngine> DEPRECATED_ENGINES = ImmutableSet.of(KNNEngine.NMSLIB);

    private static Map<KNNEngine, Integer> MAX_DIMENSIONS_BY_ENGINE = Map.of(
        KNNEngine.NMSLIB,
        16_000,
        KNNEngine.FAISS,
        16_000,
        KNNEngine.LUCENE,
        16_000
    );

    // Query-time method_parameters names contributed by registered engines
    // (KNNEngineDefinition#engineSpecificQueryParameters); empty in a default build.
    private static final Set<String> ENGINE_CONTRIBUTED_QUERY_PARAMETERS = ImmutableSet.copyOf(
        KNNEngineRegistry.engineContributedQueryParameterNames()
    );

    // All engines (in declaration order) whose library declares the capability. UNDEFINED has no library.
    private static Set<KNNEngine> enginesWith(Predicate<KNNLibrary> capability) {
        ImmutableSet.Builder<KNNEngine> builder = ImmutableSet.builder();
        for (KNNEngine engine : BY_NAME.values()) {
            if (engine.knnLibrary != null && capability.test(engine)) {
                builder.add(engine);
            }
        }
        return builder.build();
    }

    private final String enumName; // former enum-constant name (e.g. "FAISS"); preserves name()/toString()
    private final String name;
    private final KNNLibrary knnLibrary;
    private final Version restrictedFromVersion; // Nullable field
    private final NativeEngineService nativeService; // null for built-ins; set for runtime-registered engines

    private KNNEngine(
        String enumName,
        String name,
        KNNLibrary knnLibrary,
        Version restrictedFromVersion,
        NativeEngineService nativeService
    ) {
        this.enumName = enumName;
        this.name = name;
        this.knnLibrary = knnLibrary;
        this.restrictedFromVersion = restrictedFromVersion;
        this.nativeService = nativeService;
    }

    /**
     * The native index lifecycle for this engine, or {@code null} for a built-in engine whose native ops are
     * served by the core {@code FaissService}/{@code NmslibService}. A runtime-registered engine returns its own
     * {@link NativeEngineService}, which {@code JNIService} dispatches to generically.
     *
     * @return the engine's native service, or {@code null} if it is a built-in handled by the core services
     */
    @ExperimentalApi
    public NativeEngineService getNativeService() {
        return nativeService;
    }

    /**
     * The former enum-constant identifier (e.g. {@code "FAISS"}), preserved so callers and serialized output that
     * relied on the enum's {@code name()}/{@code toString()} (such as the query explanation string) are unchanged.
     *
     * @return the engine's constant-style name
     */
    public String name() {
        return enumName;
    }

    /**
     * All known engines (built-ins plus any runtime-registered engine). Mirrors the former enum {@code values()}.
     *
     * @return array of all engines
     */
    public static KNNEngine[] values() {
        return ALL.clone();
    }

    /**
     * Get the engine
     *
     * @param name of engine to be fetched
     * @return KNNEngine corresponding to name
     */
    public static KNNEngine getEngine(String name) {
        final KNNEngine engine = BY_NAME.get(name == null ? null : name.toLowerCase(Locale.ROOT));
        if (engine != null) {
            return engine;
        }
        throw new IllegalArgumentException(
            String.format(
                "Invalid engine type: %s. If an engine definition for this name exists, it may have failed to load; check startup warnings.",
                name
            )
        );
    }

    /**
     * Whether a registered engine has declared this query-time method parameter name (see
     * {@link KNNEngineDefinition#engineSpecificQueryParameters()}). The REST/gRPC parse layers use this to
     * defer — rather than reject — a name unknown to the core {@code MethodParameter} enum, so the
     * engine-aware validation in {@code KNNQueryBuilder#doToQuery} can judge it against the engine's
     * {@link KNNLibrarySearchContext}. Matching is exact (case-sensitive), mirroring
     * {@code MethodParameter.enumOf}.
     *
     * @param name the method parameter name from the query
     * @return true if a registered engine declared the name; false otherwise
     */
    @ExperimentalApi
    public static boolean isEngineContributedQueryParameter(String name) {
        return name != null && ENGINE_CONTRIBUTED_QUERY_PARAMETERS.contains(name);
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
        // Only custom-segment-file engines have a file extension; Lucene's getExtension() throws.
        for (KNNEngine engine : CUSTOM_SEGMENT_FILE_ENGINES) {
            if (path.endsWith(engine.getExtension()) || path.endsWith(engine.getCompoundExtension())) {
                return engine;
            }
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
    public String toString() {
        // Preserve the former enum behavior (toString == constant name, e.g. "FAISS").
        return enumName;
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

    @Override
    public boolean supportsIterativeBuild() {
        return knnLibrary.supportsIterativeBuild();
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        return knnLibrary.createsCustomSegmentFiles();
    }

    @Override
    public boolean supportsFilters() {
        return knnLibrary.supportsFilters();
    }

    @Override
    public boolean supportsRadialSearch() {
        return knnLibrary.supportsRadialSearch();
    }

    @Override
    public boolean supportsNestedFields() {
        return knnLibrary.supportsNestedFields();
    }

    @Override
    public RescoreContext getRescoreContext(
        CompressionLevel compression,
        Mode mode,
        int dimension,
        Version version,
        boolean isFlatMethod,
        boolean isSQOneBit
    ) {

        // Special handling for Lucene Scalar Quantizer (x32 compression)
        // Engine check is temporary until binary scalar quantizer is finalized for FAISS as well
        if (compression == CompressionLevel.x32 && this == LUCENE && version.onOrAfter(Version.V_3_6_0)) {
            return RescoreContext.builder()
                .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_DEFAULT_FOR_LUCENE_SCALAR_QUANTIZER_AFTER_V360)
                .userProvided(false)
                .build();
        } else {
            return null;
        }
    }
}
