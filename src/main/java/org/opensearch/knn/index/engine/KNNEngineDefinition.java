/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

import java.util.Set;

/**
 * Service-provider interface contributing a complete engine to the core k-NN module at runtime, discovered via
 * {@link java.util.ServiceLoader} (see {@link KNNEngineRegistry}) and wired in as a first-class
 * {@link KNNEngine}, resolved by name, with no compile-time reference to the contributing module.
 *
 * <p>An implementation supplies:
 * <ul>
 *   <li>{@link #engineName()} — the engine name users type in their mapping.</li>
 *   <li>{@link #library()} — the {@link KNNLibrary} driving method resolution, file extension, validation and
 *       scoring.</li>
 *   <li>{@link #nativeService()} — the native index lifecycle: {@code JNIService} routes the eight
 *       lifecycle/search operations (init/insert/write/template/load/query/radiusQuery/free) here; binary
 *       indexes, training and shared index state remain core-only today.</li>
 * </ul>
 *
 * <p>Registered engines currently inherit core defaults they cannot override: the default engine's maximum
 * dimension (16,000) and no version restriction.
 *
 * <p>When no definition is on the classpath (the default build) the registry is empty and the plugin is
 * byte-for-byte upstream.
 */
@ExperimentalApi
public interface KNNEngineDefinition {

    /**
     * The engine name users type in their mapping; matched case-insensitively. Must be non-null and non-empty;
     * a name colliding with a built-in engine or an already-registered definition is skipped with a warning.
     */
    String engineName();

    KNNLibrary library();

    /**
     * The engine's native index lifecycle, or {@code null} for a pure-JVM engine whose library does not
     * create custom segment files (such an engine never reaches {@code JNIService}).
     *
     * <p>Implementations must not touch {@code KNNEngine} statics during construction or from this method:
     * definitions are consulted during {@code KNNEngine}'s own class initialization, so doing so creates an init cycle.
     *
     * @return the engine's {@link NativeEngineService}, or {@code null} for a pure-JVM engine
     */
    default NativeEngineService nativeService() {
        return null;
    }

    /**
     * Query-time {@code method_parameters} names this engine contributes (for example {@code search_window_size}),
     * beyond the core-known names in {@code org.opensearch.knn.index.query.request.MethodParameter}. This is a
     * parse-time allowlist only: it tells the REST/gRPC layers not to reject the name, so the engine-aware
     * validation in {@code KNNQueryBuilder#doToQuery} (against the engine's {@link KNNLibrarySearchContext})
     * can judge the value. Names, not semantics — the search context remains the validation authority, so a
     * name declared here but absent from the search context is accepted at parse and then rejected there,
     * never silently honored.
     *
     * <p>The same class-initialization rule as {@link #nativeService()} applies: do not touch
     * {@code KNNEngine} statics from this method.
     *
     * @return the engine-specific query parameter names; empty (the default) if the engine's query-time
     *         parameters are all core-known
     */
    default Set<String> engineSpecificQueryParameters() {
        return Set.of();
    }
}
