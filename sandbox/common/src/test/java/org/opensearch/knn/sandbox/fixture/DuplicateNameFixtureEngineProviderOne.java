/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * One of two well formed definitions sharing an engine name. The registry must drop both without calling
 * initialize on either.
 */
public final class DuplicateNameFixtureEngineProviderOne implements KNNEngineDefinition {

    static volatile boolean initialized = false;

    @Override
    public String engineName() {
        return "duplicate-name";
    }

    @Override
    public KNNLibrary library() {
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public void initialize(KNNEngineContext context) {
        initialized = true;
    }
}
