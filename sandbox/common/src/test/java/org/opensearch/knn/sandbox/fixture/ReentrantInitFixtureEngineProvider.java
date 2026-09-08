/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;

/**
 * Calls back into KNNEngine from its initialize. Discovery must not recurse, the callback sees the
 * built-in engines and the engine still registers.
 */
public final class ReentrantInitFixtureEngineProvider implements KNNEngineDefinition {

    static volatile boolean sawFaissDuringInitialize = false;

    @Override
    public String engineName() {
        return "reentrant-init";
    }

    @Override
    public KNNLibrary library() {
        return PlainFixtureLibrary.INSTANCE;
    }

    @Override
    public void initialize(KNNEngineContext context) {
        sawFaissDuringInitialize = KNNEngine.getEngine("faiss") == KNNEngine.FAISS;
    }
}
