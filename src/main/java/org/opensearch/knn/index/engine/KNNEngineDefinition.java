/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.index.mapper.EngineFieldStrategy;

import java.util.Set;

/**
 * Service-provider interface contributing a complete engine to the core k-NN module at runtime, discovered via
 * {@link java.util.ServiceLoader} (see {@link KNNEngineRegistry}) and wired in as a first-class
 * {@link KNNEngine}, resolved by name, with no compile-time reference to the contributing module. When no
 * definition is on the classpath (the default build) the registry is empty and the plugin is byte-for-byte
 * upstream.
 *
 * <p>Every method is read exactly once at discovery and its result cached. Definitions may be consulted
 * during {@code KNNEngine} class initialization, so no method may touch {@code KNNEngine} statics.
 */
@ExperimentalApi
public interface KNNEngineDefinition {

    /** The engine name users type in their mapping; matched case-insensitively, must be non-blank and unique. */
    String engineName();

    /** The {@link KNNLibrary} driving method resolution, validation, scoring and file extensions. */
    KNNLibrary library();

    /**
     * The engine's native index lifecycle, routed from {@code JNIService}, or {@code null} (the default) for
     * a pure-JVM engine whose library creates no custom segment files.
     */
    default NativeEngineService nativeService() {
        return null;
    }

    /**
     * Query-time {@code method_parameters} names this engine contributes beyond the core-known ones. A
     * parse-time allowlist only: values are still validated against the engine's
     * {@link KNNLibrarySearchContext}, never silently honored.
     */
    default Set<String> engineSpecificQueryParameters() {
        return Set.of();
    }

    /**
     * The engine's field-type construction strategy (see {@code KNNEngine#getFieldStrategy}), or
     * {@code null} (the default), in which case mapping a field with this engine fails with
     * {@code UnsupportedOperationException}.
     */
    default EngineFieldStrategy fieldStrategy() {
        return null;
    }

    /**
     * Called once as the last step of discovery. A throw skips the engine with a warning. Both context
     * fields are null outside a node. Implementations must not block, node startup waits on this, and
     * engine lookups here see only the built-ins.
     */
    default void initialize(KNNEngineContext context) {}

    /**
     * Called at node shutdown in reverse registration order, only for definitions whose {@link #initialize}
     * completed; release here whatever it acquired. Best effort: a throw is logged and the remaining
     * definitions still close.
     */
    default void close() {}
}
