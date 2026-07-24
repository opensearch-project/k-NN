/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.nmslib.Nmslib;

import java.util.ArrayList;
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
 * Empty in a default build.
 */
@Log4j2
final class KNNEngineRegistry {

    // Built-in engine names, spelled out here (not read off KNNEngine) because this registry loads during
    // KNNEngine's class initialization. KNNEngine asserts its built-ins against this set when seeding.
    static final Set<String> BUILT_IN_ENGINE_NAMES = Set.of("faiss", "lucene", "nmslib", "undefined");

    // Segment-file extensions owned by built-in libraries (safe to read here: the library singletons do not
    // depend on KNNEngine's class initialization).
    private static final Set<String> BUILT_IN_ENGINE_EXTENSIONS = Set.of(Faiss.INSTANCE.getExtension(), Nmslib.INSTANCE.getExtension());

    /** A fully-materialized registered engine; every definition method has already been invoked successfully. */
    record RegisteredEngine(String engineName, KNNLibrary library, NativeEngineService nativeService, Set<String> queryParameterNames) {
    }

    private static final Map<String, RegisteredEngine> BY_NAME;
    private static final Set<String> QUERY_PARAMETER_NAMES;

    static {
        final Map<String, List<RegisteredEngine>> candidatesByName = new LinkedHashMap<>();
        for (KNNEngineDefinition definition : ServiceLoader.load(KNNEngineDefinition.class, KNNEngineRegistry.class.getClassLoader())) {
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
                final RegisteredEngine engine = new RegisteredEngine(
                    name,
                    definition.library(),
                    definition.nativeService(),
                    Set.copyOf(definition.engineSpecificQueryParameters())
                );
                if (engine.library() == null) {
                    log.warn("KNNEngineDefinition [{}] returned a null library; ignoring", definition.getClass().getName());
                    continue;
                }
                if (engine.library().createsCustomSegmentFiles() && engine.nativeService() == null) {
                    log.warn(
                        "KNNEngineDefinition [{}] creates custom segment files but supplies no NativeEngineService to serve them; ignoring",
                        definition.getClass().getName()
                    );
                    continue;
                }
                candidatesByName.computeIfAbsent(key, k -> new ArrayList<>()).add(engine);
            } catch (Exception | LinkageError e) {
                log.warn("Skipping misconfigured KNNEngineDefinition", e);
            }
        }

        // A duplicate name is always a misconfiguration; dropping every claimant keeps the outcome
        // deterministic regardless of classpath order.
        final Map<String, RegisteredEngine> byName = new LinkedHashMap<>();
        final Set<String> queryParameterNames = new HashSet<>();
        final Set<String> reservedExtensions = new HashSet<>(BUILT_IN_ENGINE_EXTENSIONS);
        for (Map.Entry<String, List<RegisteredEngine>> entry : candidatesByName.entrySet()) {
            if (entry.getValue().size() > 1) {
                log.warn("Multiple KNNEngineDefinitions register the name [{}]; ignoring all of them", entry.getKey());
                continue;
            }
            final RegisteredEngine engine = entry.getValue().get(0);
            if (engine.library().createsCustomSegmentFiles() && !claimExtension(engine, reservedExtensions)) {
                continue;
            }
            byName.put(entry.getKey(), engine);
            queryParameterNames.addAll(engine.queryParameterNames());
        }
        BY_NAME = Collections.unmodifiableMap(byName);
        QUERY_PARAMETER_NAMES = Collections.unmodifiableSet(queryParameterNames);
    }

    /**
     * A registered extension must be non-blank and must not suffix-collide with any built-in or
     * already-claimed extension, or {@code getEngineNameFromPath} would misroute segment files.
     */
    private static boolean claimExtension(RegisteredEngine engine, Set<String> reservedExtensions) {
        final String extension;
        try {
            extension = engine.library().getExtension();
        } catch (Exception | LinkageError e) {
            log.warn("Registered engine [{}]: library.getExtension() failed; ignoring the engine", engine.engineName(), e);
            return false;
        }
        if (extension == null || extension.isBlank()) {
            log.warn("Registered engine [{}] declares custom segment files but no file extension; ignoring", engine.engineName());
            return false;
        }
        for (String reserved : reservedExtensions) {
            if (extension.endsWith(reserved) || reserved.endsWith(extension)) {
                log.warn(
                    "Registered engine [{}] extension [{}] suffix-collides with [{}]; ignoring the engine",
                    engine.engineName(),
                    extension,
                    reserved
                );
                return false;
            }
        }
        reservedExtensions.add(extension);
        return true;
    }

    private KNNEngineRegistry() {}

    /** All registered engines discovered on the classpath (empty in a default build). */
    static Collection<RegisteredEngine> all() {
        return BY_NAME.values();
    }

    /**
     * Query-time {@code method_parameters} names contributed by registered engines (see
     * {@link KNNEngineDefinition#engineSpecificQueryParameters()}); empty in a default build.
     */
    static Set<String> engineContributedQueryParameterNames() {
        return QUERY_PARAMETER_NAMES;
    }
}
