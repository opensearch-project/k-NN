/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import org.opensearch.knn.index.query.KNNQueryResult;

public abstract class ResultsCollector {
    public abstract void addResult(int docId, float distance);

    public abstract KNNQueryResult[] getResults();
}
