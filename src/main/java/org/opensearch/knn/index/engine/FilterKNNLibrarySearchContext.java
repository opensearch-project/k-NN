/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Map;

@AllArgsConstructor
public abstract class FilterKNNLibrarySearchContext implements KNNLibrarySearchContext {
    private final KNNLibrarySearchContext delegate;

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return delegate.supportedMethodParameters(ctx);
    }

    @Override
    public Map<String, Object> processMethodParameters(QueryContext ctx, Map<String, Object> parameters) {
        return delegate.processMethodParameters(ctx, parameters);
    }

    @Override
    public RescoreContext getDefaultRescoreContext(QueryContext ctx) {
        return delegate.getDefaultRescoreContext(ctx);
    }
}
