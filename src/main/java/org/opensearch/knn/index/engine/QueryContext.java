/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import org.opensearch.knn.index.VectorQueryType;

/**
 * Context class for query-specific information.
 */
@AllArgsConstructor
public class QueryContext {
    VectorQueryType queryType;
}
