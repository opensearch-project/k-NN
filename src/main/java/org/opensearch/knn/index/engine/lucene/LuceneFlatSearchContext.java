/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.util.Collections;
import java.util.Map;

/**
 * Search context for the Lucene flat (BBQ) method. The flat method performs
 * exact nearest-neighbor search using 1-bit scalar quantization and does not
 * expose any search-time parameters.
 */
public class LuceneFlatSearchContext implements KNNLibrarySearchContext {

    /**
     * Returns an empty map because the flat method has no configurable
     * search-time parameters.
     *
     * @param ctx the query context, unused
     * @return an empty map
     */
    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return Collections.emptyMap();
    }
}
