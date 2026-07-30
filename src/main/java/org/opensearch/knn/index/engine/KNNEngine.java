/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableSet;
import lombok.extern.log4j.Log4j2;
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

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.UNDEFINED_ENGINE_NAME;

/**
 * KNNEngine provides the functionality to validate and transform user defined indices into information that can be
 * passed to the respective k-NN library's JNI layer.
 *
 * <p>This was a Java {@code enum} with a fixed set of constants and is now an open, instance based type. The
 * built-in engines are still {@code public static final} singletons, so existing references and identity
 * checks are unchanged, and more engines can be contributed at runtime through {@link KNNEngineDefinition}.
 * The public surface is preserved.
 */
@Log4j2
public final class KNNEngine implements KNNLibrary, VectorSearchEngine {

    // Built-ins pass a null native service because their JNI lifecycle is the core FaissService and
    // NmslibService. A runtime registered engine carries its own NativeEngineService.
    @Deprecated(since = "2.19.0", forRemoval = true)
    public static final KNNEngine NMSLIB = new KNNEngine("NMSLIB", NMSLIB_NAME, Nmslib.INSTANCE, Version.V_3_0_0);
    public static final KNNEngine FAISS = new KNNEngine("FAISS", FAISS_NAME, Faiss.INSTANCE);
    public static final KNNEngine LUCENE = new KNNEngine("LUCENE", LUCENE_NAME, Lucene.INSTANCE);
    public static final KNNEngine UNDEFINED = new KNNEngine("UNDEFINED", UNDEFINED_ENGINE_NAME, null);

    public static final KNNEngine DEFAULT = FAISS;

    // Built-in engines in declaration order. Registered engines are appended when discovery runs.
    private static final List<KNNEngine> BUILT_INS = List.of(NMSLIB, FAISS, LUCENE, UNDEFINED);

    /** One immutable snapshot of built-ins plus every registered engine, swapped once when discovery runs. */
    private record EngineTable(Map<String, KNNEngine> byName, KNNEngine[] all, Set<KNNEngine> customSegmentFileEngines, Set<
        String> engineContributedQueryParameters) {

        static EngineTable of(Map<String, KNNEngine> byName, Set<String> engineContributedQueryParameters) {
            final ImmutableSet.Builder<KNNEngine> customFiles = ImmutableSet.builder();
            for (KNNEngine engine : byName.values()) {
                if (engine.createsCustomSegmentFiles()) {
                    customFiles.add(engine);
                }
            }
            return new EngineTable(
                java.util.Collections.unmodifiableMap(byName),
                byName.values().toArray(new KNNEngine[0]),
                customFiles.build(),
                Set.copyOf(engineContributedQueryParameters)
            );
        }
    }

    // Discovery runs from KNNPlugin#createComponents in production, which hands definitions the node
    // services. Tests and tools that never construct the plugin discover lazily on first use.
    private static volatile EngineTable TABLE = EngineTable.of(builtInsByName(), Set.of());
    private static volatile boolean DISCOVERY_ATTEMPTED = false;
    // True only while a discovery pass runs. Volatile so a lookup during the pass serves the built-ins
    // snapshot instead of waiting on the lock, and so a definition that calls back into KNNEngine from
    // its initialize returns without recursing.
    private static volatile boolean DISCOVERING = false;

    /**
     * Discovers and registers runtime engines, handing each definition the given context. Idempotent, the
     * first caller wins; called by {@code KNNPlugin#createComponents} with real node services.
     */
    public static void initialize(KNNEngineContext context) {
        synchronized (KNNEngine.class) {
            if (DISCOVERY_ATTEMPTED) {
                if (context != KNNEngineContext.EMPTY) {
                    log.warn("Node services arrived after engine discovery had already run; engine definitions did not receive them");
                }
                return;
            }
            if (DISCOVERING) {
                // A definition called back into KNNEngine from its initialize. It sees the built-ins.
                return;
            }
            if (context == KNNEngineContext.EMPTY) {
                log.info("Engine discovery running without node services; expected only outside a node (tests and tools)");
            }
            DISCOVERING = true;
            // The finally latches even if the pass failed, so lookups never rerun discovery and
            // initialize stays once only.
            try {
                final KNNEngineRegistry.DiscoveryResult discovery = KNNEngineRegistry.discover(context);
                final Map<String, KNNEngine> byName = builtInsByName();
                for (KNNEngineRegistry.RegisteredEngine registered : discovery.engines()) {
                    byName.put(
                        registered.engineName().toLowerCase(Locale.ROOT),
                        new KNNEngine(
                            registered.engineName().toUpperCase(Locale.ROOT),
                            registered.engineName(),
                            registered.library(),
                            registered.nativeService(),
                            registered.capabilities(),
                            registered.extension(),
                            registered.compoundExtension()
                        )
                    );
                }
                TABLE = EngineTable.of(byName, discovery.queryParameterNames());
            } finally {
                DISCOVERING = false;
                DISCOVERY_ATTEMPTED = true;
            }
        }
    }

    private static Map<String, KNNEngine> builtInsByName() {
        final Map<String, KNNEngine> byName = new LinkedHashMap<>();
        for (KNNEngine engine : BUILT_INS) {
            byName.put(engine.name.toLowerCase(Locale.ROOT), engine);
        }
        return byName;
    }

    private static EngineTable table() {
        // While a discovery pass is in flight every lookup serves the current snapshot, nothing blocks.
        if (DISCOVERY_ATTEMPTED == false && DISCOVERING == false) {
            initialize(KNNEngineContext.EMPTY);
        }
        return TABLE;
    }

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

    private final String enumName; // former enum-constant name (e.g. "FAISS"); preserves name()/toString()
    private final String name;
    private final KNNLibrary knnLibrary;
    private final Version restrictedFromVersion; // Nullable field
    private final NativeEngineService nativeService; // null for built-ins; set for runtime-registered engines
    // For a registered engine the capability flags and the segment file extensions are read once at
    // discovery and cached here, so plugin code is never asked again for them. Null for built-ins, which
    // delegate to their core library.
    private final KNNEngineRegistry.EngineCapabilities capabilities;
    private final String extension;
    private final String compoundExtension;

    private KNNEngine(String enumName, String name, KNNLibrary knnLibrary) {
        this(enumName, name, knnLibrary, null, null, null, null, null);
    }

    private KNNEngine(String enumName, String name, KNNLibrary knnLibrary, Version restrictedFromVersion) {
        this(enumName, name, knnLibrary, restrictedFromVersion, null, null, null, null);
    }

    // Runtime-registered engines only.
    private KNNEngine(
        String enumName,
        String name,
        KNNLibrary knnLibrary,
        NativeEngineService nativeService,
        KNNEngineRegistry.EngineCapabilities capabilities,
        String extension,
        String compoundExtension
    ) {
        this(enumName, name, knnLibrary, null, nativeService, capabilities, extension, compoundExtension);
    }

    private KNNEngine(
        String enumName,
        String name,
        KNNLibrary knnLibrary,
        Version restrictedFromVersion,
        NativeEngineService nativeService,
        KNNEngineRegistry.EngineCapabilities capabilities,
        String extension,
        String compoundExtension
    ) {
        this.enumName = enumName;
        this.name = name;
        this.knnLibrary = knnLibrary;
        this.restrictedFromVersion = restrictedFromVersion;
        this.nativeService = nativeService;
        this.capabilities = capabilities;
        this.extension = extension;
        this.compoundExtension = compoundExtension;
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
        return table().all().clone();
    }

    /**
     * Get the engine
     *
     * @param name of engine to be fetched
     * @return KNNEngine corresponding to name
     */
    public static KNNEngine getEngine(String name) {
        final KNNEngine engine = table().byName().get(name == null ? null : name.toLowerCase(Locale.ROOT));
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
        return name != null && table().engineContributedQueryParameters().contains(name);
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
        for (KNNEngine engine : table().customSegmentFileEngines()) {
            if (path.endsWith(engine.getExtension()) || path.endsWith(engine.getCompoundExtension())) {
                return engine;
            }
        }
        throw new IllegalArgumentException("No engine matches the path's suffix");
    }

    /**
     * Returns all engines that create custom segment files. The one collection accessor kept, because its
     * callers iterate the set and a test mocks it.
     *
     * @return Set of all engines that create custom segment files.
     */
    public static Set<KNNEngine> getEnginesThatCreateCustomSegmentFiles() {
        return table().customSegmentFileEngines();
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
        if (extension != null) {
            return extension;
        }
        return knnLibrary.getExtension();
    }

    @Override
    public String getCompoundExtension() {
        if (compoundExtension != null) {
            return compoundExtension;
        }
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
        if (capabilities != null) {
            return capabilities.iterativeBuild();
        }
        return knnLibrary != null && knnLibrary.supportsIterativeBuild();
    }

    @Override
    public boolean createsCustomSegmentFiles() {
        if (capabilities != null) {
            return capabilities.customSegmentFiles();
        }
        return knnLibrary != null && knnLibrary.createsCustomSegmentFiles();
    }

    @Override
    public boolean supportsFilters() {
        if (capabilities != null) {
            return capabilities.filters();
        }
        return knnLibrary != null && knnLibrary.supportsFilters();
    }

    @Override
    public boolean supportsRadialSearch() {
        if (capabilities != null) {
            return capabilities.radialSearch();
        }
        return knnLibrary != null && knnLibrary.supportsRadialSearch();
    }

    @Override
    public boolean supportsNestedFields() {
        if (capabilities != null) {
            return capabilities.nestedFields();
        }
        return knnLibrary != null && knnLibrary.supportsNestedFields();
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
