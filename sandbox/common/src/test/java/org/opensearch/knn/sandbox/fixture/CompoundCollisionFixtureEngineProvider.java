/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineContext;
import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

/**
 * Well formed except its extension is faiss's compound extension. The registry must reject it at the
 * claim, so its initialize never runs, or faiss compound segment files would route to it.
 */
public final class CompoundCollisionFixtureEngineProvider implements KNNEngineDefinition {

    static volatile boolean initialized = false;

    private static final KNNLibrary LIBRARY = new PlainFixtureLibrary(".faissc") {
        @Override
        public boolean createsCustomSegmentFiles() {
            return true;
        }
    };

    private static final NativeEngineService SERVICE = new AbstractNativeEngineService("compound-collision") {
    };

    @Override
    public String engineName() {
        return "compound-collision";
    }

    @Override
    public KNNLibrary library() {
        return LIBRARY;
    }

    @Override
    public NativeEngineService nativeService() {
        return SERVICE;
    }

    @Override
    public void initialize(KNNEngineContext context) {
        initialized = true;
    }
}
