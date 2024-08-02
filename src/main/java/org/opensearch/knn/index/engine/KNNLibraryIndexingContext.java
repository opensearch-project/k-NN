/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import java.util.Collections;
import java.util.Map;

/**
 * Context a library gives to build one of its indices
 */
public interface KNNLibraryIndexingContext {
    /**
     * Get map of parameters that get passed to the library to build the index
     *
     * @return Map of parameters
     */
    Map<String, Object> getLibraryParameters();

    KNNLibraryIndexingContext EMPTY = Collections::emptyMap;
}
