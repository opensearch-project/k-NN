/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.KNNEngineDefinition;
import org.opensearch.knn.index.engine.KNNLibrary;
import org.opensearch.knn.index.engine.NativeEngineService;
import org.opensearch.knn.index.mapper.EngineFieldStrategy;
import org.opensearch.knn.index.mapper.FaissFieldStrategy;

import java.util.Set;

/**
 * {@link KNNEngineDefinition} for the experimental Intel SVS engine, discovered via {@code META-INF/services}.
 * Discovery never loads the native library; it loads lazily on the first native call.
 */
public class SvsEngineProvider implements KNNEngineDefinition {

    private final NativeEngineService nativeService = new SvsNativeEngineService();

    @Override
    public String engineName() {
        return SVSConstants.SVS_ENGINE_NAME;
    }

    @Override
    public KNNLibrary library() {
        return SvsLibrary.INSTANCE;
    }

    @Override
    public NativeEngineService nativeService() {
        return nativeService;
    }

    @Override
    public EngineFieldStrategy fieldStrategy() {
        // SVS indexes are faiss-format and score like faiss.
        return FaissFieldStrategy.INSTANCE;
    }

    @Override
    public Set<String> engineSpecificQueryParameters() {
        return Set.of(SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE, SVSConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY);
    }
}
