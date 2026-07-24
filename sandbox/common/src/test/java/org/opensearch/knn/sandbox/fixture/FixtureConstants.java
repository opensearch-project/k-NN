/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

/**
 * Names used by the test-only fixture engine, which exercises the generic engine extension points in CI
 * without native code.
 */
public final class FixtureConstants {

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

    private FixtureConstants() {}
}
