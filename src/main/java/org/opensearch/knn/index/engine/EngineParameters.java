/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The engine specific parameters of one operation, read through declared {@link ParameterKey}s. The
 * collection stays open because each engine defines its own keys, the wire and the parse layers keep
 * carrying plain maps, and {@link #raw()} exposes the map for code that needs it.
 */
@ExperimentalApi
public final class EngineParameters {

    public static final EngineParameters EMPTY = new EngineParameters(Map.of());

    private final Map<String, Object> raw;

    private EngineParameters(Map<String, Object> raw) {
        this.raw = raw;
    }

    public static EngineParameters of(Map<String, ?> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return EMPTY;
        }
        return new EngineParameters(Collections.unmodifiableMap(new LinkedHashMap<>(parameters)));
    }

    /** The value for the key, or null when absent. Throws when the value does not match the key's type. */
    public <T> T get(ParameterKey<T> key) {
        return key.cast(raw.get(key.name()));
    }

    /** The value for the key, or the default when absent. */
    public <T> T get(ParameterKey<T> key, T defaultValue) {
        final T value = get(key);
        return value == null ? defaultValue : value;
    }

    public boolean has(ParameterKey<?> key) {
        return raw.containsKey(key.name());
    }

    public Map<String, Object> raw() {
        return raw;
    }
}
