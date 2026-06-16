/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Map;

import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;

/**
 * Faiss SVS Vamana search context. The only query-time tunable is {@code search_window_size} (the
 * recall/latency knob, analogous to {@code ef_search}); {@code search_buffer_capacity} is an index-time
 * parameter only.
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana search context", since = "3.7.0")
public final class FaissSVSVamanaSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(
            METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, null, (v, context) -> v > 0)
        );
    }
}
