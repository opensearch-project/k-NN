/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.engine.faiss.Faiss;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.UNDEFINED_ENGINE_NAME;
import org.opensearch.knn.index.engine.nmslib.Nmslib;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.ServiceConfigurationError;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Discovers engines contributed at runtime via {@link java.util.ServiceLoader} of {@link KNNEngineDefinition}.
 * A definition that throws or is misconfigured is skipped with a warning rather than failing the plugin.
 * Empty in a default build. Discovery runs once per JVM, so when several nodes share one JVM, as in
 * integration test clusters, only the first node's services reach the definitions.
 */
@Log4j2
final class KNNEngineRegistry {

    // Shared KNNConstants names, not KNNEngine fields. Discovery can run from the plugin lifecycle or
    // lazily during KNNEngine class initialization, and reading KNNEngine here would cycle in the lazy case.
    static final Set<String> BUILT_IN_ENGINE_NAMES = Set.of(FAISS_NAME, LUCENE_NAME, NMSLIB_NAME, UNDEFINED_ENGINE_NAME);

    // Segment-file extensions owned by built-in libraries, plain and compound forms (safe to read here:
    // the library singletons do not depend on KNNEngine's class initialization).
    private static final Set<String> ENGINE_SEGMENT_FILES_EXTENSIONS = Set.of(
        Faiss.INSTANCE.getExtension(),
        Faiss.INSTANCE.getCompoundExtension(),
        Nmslib.INSTANCE.getExtension(),
        Nmslib.INSTANCE.getCompoundExtension()
    );

    /** The five capability flags, read once from the library during validation. */
    record EngineCapabilities(boolean iterativeBuild, boolean customSegmentFiles, boolean filters, boolean radialSearch,
        boolean nestedFields) {
    }

    /**
     * A fully-materialized registered engine; every definition method has already been invoked successfully.
     * The capabilities and extension are read once during validation and cached here, so no later step has
     * to call back into plugin code.
     */
    record RegisteredEngine(String engineName, KNNLibrary library, NativeEngineService nativeService, Set<String> queryParameterNames,
        EngineCapabilities capabilities, String extension, String compoundExtension) {
    }

    /** A validated candidate plus its definition, so initialize runs only for engines that register. */
    private record Candidate(KNNEngineDefinition definition, RegisteredEngine engine) {
    }

    /** The outcome of one discovery pass over the classpath. */
    record DiscoveryResult(Collection<RegisteredEngine> engines, Set<String> queryParameterNames) {
    }

    /**
     * Discovers, validates and materializes every {@link KNNEngineDefinition} on the classpath. Invoked by
     * {@code KNNEngine} when discovery is triggered (from the plugin lifecycle in production, lazily in
     * tests and tools). Empty result in a default build.
     */
    static DiscoveryResult discover(KNNEngineContext context) {
        final Map<String, List<Candidate>> candidatesByName = new LinkedHashMap<>();
        final Iterator<KNNEngineDefinition> definitions = ServiceLoader.load(
            KNNEngineDefinition.class,
            KNNEngineRegistry.class.getClassLoader()
        ).iterator();
        while (true) {
            final KNNEngineDefinition definition;
            try {
                if (definitions.hasNext() == false) {
                    break;
                }
                definition = definitions.next();
            } catch (ServiceConfigurationError | LinkageError e) {
                // A provider class that fails to load or construct must not take the node down.
                log.warn("Skipping a KNNEngineDefinition provider that failed to load", e);
                continue;
            }
            try {
                final String name = definition.engineName();
                if (name == null || name.isBlank()) {
                    log.warn("KNNEngineDefinition [{}] returned a null or blank engine name; ignoring", definition.getClass().getName());
                    continue;
                }
                final String key = name.toLowerCase(Locale.ROOT);
                if (BUILT_IN_ENGINE_NAMES.contains(key)) {
                    log.warn(
                        "KNNEngineDefinition [{}] collides with built-in engine name [{}]; ignoring",
                        definition.getClass().getName(),
                        key
                    );
                    continue;
                }
                final KNNLibrary library = definition.library();
                if (library == null) {
                    log.warn("KNNEngineDefinition [{}] returned a null library; ignoring", definition.getClass().getName());
                    continue;
                }
                final EngineCapabilities capabilities = new EngineCapabilities(
                    library.supportsIterativeBuild(),
                    library.createsCustomSegmentFiles(),
                    library.supportsFilters(),
                    library.supportsRadialSearch(),
                    library.supportsNestedFields()
                );
                final RegisteredEngine engine = new RegisteredEngine(
                    name,
                    library,
                    definition.nativeService(),
                    Set.copyOf(definition.engineSpecificQueryParameters()),
                    capabilities,
                    capabilities.customSegmentFiles() ? library.getExtension() : null,
                    capabilities.customSegmentFiles() ? library.getCompoundExtension() : null
                );
                if (engine.capabilities().customSegmentFiles() && engine.nativeService() == null) {
                    log.warn(
                        "KNNEngineDefinition [{}] creates custom segment files but supplies no NativeEngineService to serve them; ignoring",
                        definition.getClass().getName()
                    );
                    continue;
                }
                candidatesByName.computeIfAbsent(key, k -> new ArrayList<>()).add(new Candidate(definition, engine));
            } catch (Exception | LinkageError e) {
                log.warn("Skipping misconfigured KNNEngineDefinition", e);
            }
        }

        // A duplicate name is always a misconfiguration; dropping every claimant keeps the outcome
        // deterministic regardless of classpath order.
        final Map<String, RegisteredEngine> byName = new LinkedHashMap<>();
        final Set<String> queryParameterNames = new HashSet<>();
        final Set<String> reservedExtensions = new HashSet<>(ENGINE_SEGMENT_FILES_EXTENSIONS);
        for (Map.Entry<String, List<Candidate>> entry : candidatesByName.entrySet()) {
            if (entry.getValue().size() > 1) {
                log.warn("Multiple KNNEngineDefinitions register the name [{}]; ignoring all of them", entry.getKey());
                continue;
            }
            final Candidate candidate = entry.getValue().get(0);
            final RegisteredEngine engine = candidate.engine();
            if (engine.capabilities().customSegmentFiles() && !claimExtension(engine, reservedExtensions)) {
                continue;
            }
            // The engine is accepted. Hand its definition the node services now, so initialize runs exactly
            // once and only for engines that actually register. A throw here skips the engine.
            try {
                candidate.definition().initialize(context);
            } catch (Exception | LinkageError e) {
                log.warn("Registered engine [{}]: initialize failed; ignoring the engine", engine.engineName(), e);
                if (engine.capabilities().customSegmentFiles()) {
                    // Release the claim so the failed engine cannot block a later engine from the extension.
                    reservedExtensions.remove(engine.extension());
                    reservedExtensions.remove(engine.compoundExtension());
                }
                continue;
            }
            byName.put(entry.getKey(), engine);
            queryParameterNames.addAll(engine.queryParameterNames());
        }
        return new DiscoveryResult(Collections.unmodifiableCollection(byName.values()), Collections.unmodifiableSet(queryParameterNames));
    }

    /**
     * A registered extension must be non-blank, and neither its plain nor its compound form may
     * suffix-collide with any built-in or already-claimed form, or {@code getEngineNameFromPath} would
     * misroute segment files.
     */
    private static boolean claimExtension(RegisteredEngine engine, Set<String> reservedExtensions) {
        final String extension = engine.extension();
        final String compoundExtension = engine.compoundExtension();
        if (extension == null || extension.isBlank() || compoundExtension == null || compoundExtension.isBlank()) {
            log.warn("Registered engine [{}] declares custom segment files but no file extension; ignoring", engine.engineName());
            return false;
        }
        for (String candidate : List.of(extension, compoundExtension)) {
            for (String reserved : reservedExtensions) {
                if (candidate.endsWith(reserved) || reserved.endsWith(candidate)) {
                    log.warn(
                        "Registered engine [{}] extension [{}] suffix-collides with [{}]; ignoring the engine",
                        engine.engineName(),
                        candidate,
                        reserved
                    );
                    return false;
                }
            }
        }
        reservedExtensions.add(extension);
        reservedExtensions.add(compoundExtension);
        return true;
    }

    private KNNEngineRegistry() {}
}
