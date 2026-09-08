/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.ParameterKey;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Names used by the test-only fixture engine, which exercises the generic engine extension points in CI
 * without native code.
 */
public final class FixtureConstants {

    /**
     * Engine names appended by each fixture definition's {@code close()}, in call order. The lifecycle
     * tests assert reverse-registration ordering and that a throwing close does not stop the rest.
     */
    public static final List<String> CLOSE_ORDER = new CopyOnWriteArrayList<>();

    /** Engine name in mappings and {@code KNNEngine.getEngine(name)}. */
    public static final String FIXTURE_ENGINE_NAME = "sandbox_fixture";

    /** Name of the deliberately broken provider used by registration tests. */
    public static final String BAD_FIXTURE_ENGINE_NAME = "bad_fixture";

    /** The fixture library's sole method. */
    public static final String METHOD_FIXTURE = "fixture_method";

    /** File extension of the fixture's custom segment files. */
    public static final String FIXTURE_EXTENSION = ".fixturebin";

    /**
     * The fixture's engine-specific query-time parameter. Deliberately NOT in the core
     * {@code MethodParameter} enum: it exists to prove that a parameter only the engine knows about is
     * deferred by the REST/gRPC layers and carried by the generic node-to-node wire.
     */
    public static final String METHOD_PARAMETER_FIXTURE_WINDOW = "fixture_window";

    /** Typed key for the same parameter, the read side of the structured params demo. */
    public static final ParameterKey<Integer> FIXTURE_WINDOW = ParameterKey.intKey(METHOD_PARAMETER_FIXTURE_WINDOW);

    private FixtureConstants() {}
}
