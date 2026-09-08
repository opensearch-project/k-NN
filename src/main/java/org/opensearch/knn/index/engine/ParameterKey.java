/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

import java.util.Locale;

/**
 * A typed name for one engine parameter, declared once by the engine that owns it and used for typed
 * reads through {@link EngineParameters}. The key carries the type, so read sites need no casts.
 *
 * @param <T> the parameter's value type
 */
@ExperimentalApi
public final class ParameterKey<T> {

    private final String name;
    private final Class<T> type;

    private ParameterKey(String name, Class<T> type) {
        this.name = name;
        this.type = type;
    }

    public static ParameterKey<Integer> intKey(String name) {
        return new ParameterKey<>(name, Integer.class);
    }

    public static ParameterKey<Long> longKey(String name) {
        return new ParameterKey<>(name, Long.class);
    }

    public static ParameterKey<Float> floatKey(String name) {
        return new ParameterKey<>(name, Float.class);
    }

    public static ParameterKey<Boolean> boolKey(String name) {
        return new ParameterKey<>(name, Boolean.class);
    }

    public static ParameterKey<String> stringKey(String name) {
        return new ParameterKey<>(name, String.class);
    }

    public String name() {
        return name;
    }

    T cast(Object value) {
        if (value == null) {
            return null;
        }
        // Parsed request values arrive as the narrowest Number the parser chose, widen them here.
        if (value instanceof Number number) {
            if (type == Integer.class) {
                return type.cast(number.intValue());
            }
            if (type == Long.class) {
                return type.cast(number.longValue());
            }
            if (type == Float.class) {
                return type.cast(number.floatValue());
            }
        }
        if (type.isInstance(value)) {
            return type.cast(value);
        }
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "Parameter [%s] expected [%s] but was [%s]",
                name,
                type.getSimpleName(),
                value.getClass().getSimpleName()
            )
        );
    }
}
