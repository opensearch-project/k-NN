/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Collections;
import java.util.Map;

/**
 * Holds the context needed to search a knn library.
 */
public interface KNNLibrarySearchContext {

    /**
     * Returns supported parameters for the library.
     *
     * @param ctx QueryContext
     * @return parameters supported by the library
     */
    Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx);

    Map<String, Object> processMethodParameters(QueryContext ctx, Map<String, Object> parameters);

    RescoreContext getDefaultRescoreContext(QueryContext ctx);

    KNNLibrarySearchContext EMPTY = new KNNLibrarySearchContext() {
        @Override
        public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
            return Collections.emptyMap();
        }

        @Override
        public Map<String, Object> processMethodParameters(QueryContext ctx, Map<String, Object> parameters) {
            return Collections.emptyMap();
        }

        @Override
        public RescoreContext getDefaultRescoreContext(QueryContext ctx) {
            return null;
        }
    };
}
