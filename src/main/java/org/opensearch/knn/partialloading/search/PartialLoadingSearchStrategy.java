/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import org.opensearch.knn.index.query.KNNQueryResult;

public abstract class PartialLoadingSearchStrategy {
    public abstract KNNQueryResult[] queryBinaryIndex(PartialLoadingSearchParameters searchParameters);

    public abstract KNNQueryResult[] queryIndex(PartialLoadingSearchParameters searchParameters);
}
