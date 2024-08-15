/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Map;

/**
 * Default HNSW context for all engines. Have a different implementation if engine context differs.
 */
public final class DefaultHnswSearchContext implements KNNLibrarySearchContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(
            MethodParameter.EF_SEARCH.getName(),
            new Parameter.IntegerParameter(MethodParameter.EF_SEARCH.getName(), null, (value, context) -> true)
        )
        .build();

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return supportedMethodParameters;
    }
}
