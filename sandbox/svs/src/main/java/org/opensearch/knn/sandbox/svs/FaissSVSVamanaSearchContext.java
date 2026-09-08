/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.util.Map;

import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_SEARCH_WINDOW_SIZE;

/**
 * Query-time parameters for {@code svs_vamana}: {@code search_window_size} and {@code search_buffer_capacity}.
 */
public final class FaissSVSVamanaSearchContext implements KNNLibrarySearchContext {

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Map.of(
            METHOD_PARAMETER_SEARCH_WINDOW_SIZE,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_WINDOW_SIZE, null, (v, context) -> v > 0),
            METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY,
            new Parameter.IntegerParameter(METHOD_PARAMETER_SEARCH_BUFFER_CAPACITY, null, (v, context) -> v > 0)
        );
    }
}
