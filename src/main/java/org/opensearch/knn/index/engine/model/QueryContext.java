/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.knn.index.VectorQueryType;

/**
 * Context class for query-specific information.
 */
@AllArgsConstructor
@Getter
public class QueryContext {
    VectorQueryType queryType;
}
