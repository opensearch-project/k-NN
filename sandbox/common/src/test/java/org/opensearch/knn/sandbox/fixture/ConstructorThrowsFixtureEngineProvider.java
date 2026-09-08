/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * A provider whose constructor throws, so ServiceLoader fails while instantiating it. Listed first in the
 * test services file: every definition after it registering proves the registry skips a provider that
 * fails to load instead of letting it take the node down.
 */
public final class ConstructorThrowsFixtureEngineProvider implements KNNEngineDefinition {

    public ConstructorThrowsFixtureEngineProvider() {
        throw new IllegalStateException("deliberately failing provider construction");
    }

    @Override
    public String engineName() {
        return "constructor-throws";
    }

    @Override
    public KNNLibrary library() {
        return null;
    }
}
