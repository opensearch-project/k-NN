/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;

import java.util.Set;

/**
 * {@link KNNEngineDefinition} for the test-only fixture engine, discovered by the core
 * {@code KNNEngineRegistry} via the {@code META-INF/services} entry in the sandbox TEST resources — so the
 * fixture registers only on the sandbox test classpath and can never appear in a shipped artifact.
 *
 * <p>A real tenant's provider looks like this, in the tenant's main sources.
 */
public final class FixtureEngineProvider implements KNNEngineDefinition {

    @Override
    public String engineName() {
        return FixtureConstants.FIXTURE_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return FixtureLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return FixtureNativeEngineService.INSTANCE;
    }

    @Override
    public Set<String> engineSpecificQueryParameters() {
        return Set.of(FixtureConstants.METHOD_PARAMETER_FIXTURE_WINDOW);
    }
}
