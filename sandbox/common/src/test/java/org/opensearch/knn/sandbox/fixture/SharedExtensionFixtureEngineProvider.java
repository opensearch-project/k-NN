/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.fixture;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.sandbox.AbstractNativeEngineService;

/**
 * The healthy second claimant of the shared extension, see
 * {@link SharedExtensionInitThrowsFixtureEngineProvider}. It must register.
 */
public final class SharedExtensionFixtureEngineProvider implements KNNEngineDefinition {

    private static final KNNLibrary LIBRARY = new PlainFixtureLibrary(SharedExtensionInitThrowsFixtureEngineProvider.SHARED_EXTENSION) {
        @Override
        public boolean createsCustomSegmentFiles() {
            return true;
        }
    };

    private static final NativeEngineService SERVICE = new AbstractNativeEngineService("shared-extension-winner") {
    };

    @Override
    public String engineName() {
        return "shared-extension-winner";
    }

    @Override
    public KNNLibrary library() {
        return LIBRARY;
    }

    @Override
    public NativeEngineService nativeService() {
        return SERVICE;
    }
}
