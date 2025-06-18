/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.engine.model.QueryContext;

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

    /// TODO: insert the matrix here
    Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx);

    KNNLibrarySearchContext EMPTY = ctx -> Collections.emptyMap();
}
