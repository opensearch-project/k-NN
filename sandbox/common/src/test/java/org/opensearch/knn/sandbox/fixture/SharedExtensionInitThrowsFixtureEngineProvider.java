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
 * First of a pair sharing a segment file extension, listed first in the services file. Its initialize
 * throws, so the registry must release its extension claim and let the second engine register.
 */
public final class SharedExtensionInitThrowsFixtureEngineProvider implements KNNEngineDefinition {

    static final String SHARED_EXTENSION = ".sharedbin";

    private static final KNNLibrary LIBRARY = new PlainFixtureLibrary(SHARED_EXTENSION) {
        @Override
        public boolean createsCustomSegmentFiles() {
            return true;
        }
    };

    private static final NativeEngineService SERVICE = new AbstractNativeEngineService("shared-extension-loser") {
    };

    @Override
    public String engineName() {
        return "shared-extension-loser";
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
        throw new IllegalStateException("deliberately failing engine initialization");
    }
}
