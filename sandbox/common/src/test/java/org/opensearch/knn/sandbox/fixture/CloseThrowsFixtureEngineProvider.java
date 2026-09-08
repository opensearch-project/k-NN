/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * A well-formed definition whose {@code close} records itself and then throws. Registered last in the
 * services file, so under reverse-registration close order it closes first and the lifecycle test proves
 * the throw does not stop the definitions after it.
 */
public final class CloseThrowsFixtureEngineProvider implements KNNEngineDefinition {

    public static final String CLOSE_THROWS_ENGINE_NAME = "close-throws";

    @Override
    public String engineName() {
        return CLOSE_THROWS_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public void close() {
        FixtureConstants.CLOSE_ORDER.add(engineName());
        throw new RuntimeException("deliberate close failure from the close-throws fixture");
    }
}
