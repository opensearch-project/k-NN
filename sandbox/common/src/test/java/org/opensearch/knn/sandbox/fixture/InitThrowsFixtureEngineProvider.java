/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * A well-formed definition whose {@code initialize} throws. It must be skipped with a warning and never
 * appear as a registered engine.
 */
public final class InitThrowsFixtureEngineProvider implements KNNEngineDefinition {

    @Override
    public String engineName() {
        return "init-throws";
    }

    @Override
    public KNNLibrary library() {
        // Pure JVM library. FixtureLibrary creates custom segment files, and without a native service the
        // definition would be dropped at validation and initialize would never run.
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public void initialize(KNNEngineContext context) {
        throw new IllegalStateException("deliberately failing engine initialization");
    }
}
