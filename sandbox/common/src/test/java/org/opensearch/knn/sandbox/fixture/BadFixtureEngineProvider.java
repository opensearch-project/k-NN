/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * A deliberately broken {@link KNNEngineDefinition} (its {@link #library()} throws), proving the registry
 * skips a misbehaving definition instead of failing engine registration or node startup.
 */
public final class BadFixtureEngineProvider implements KNNEngineDefinition {

    @Override
    public String engineName() {
        return FixtureConstants.BAD_FIXTURE_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        throw new RuntimeException("simulated broken tenant");
    }
}
