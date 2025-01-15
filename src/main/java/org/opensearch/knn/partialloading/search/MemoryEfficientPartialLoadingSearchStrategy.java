/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.partialloading.PartialLoadingContext;
import org.opensearch.knn.partialloading.faiss.FaissIndex;

import java.io.IOException;

public class MemoryEfficientPartialLoadingSearchStrategy extends PartialLoadingSearchStrategy {
    @Override
    public KNNQueryResult[] queryBinaryIndex(PartialLoadingSearchParameters searchParameters) {
        throw new UnsupportedOperationException("Partial loading does not support vector search on binary index.");
    }

    @Override
    public KNNQueryResult[] queryIndex(PartialLoadingSearchParameters searchParameters) {
        // Prepare search
        PartialLoadingContext partialLoadingContext = searchParameters.getPartialLoadingContext();
        FaissIndex faissIndex = partialLoadingContext.getFaissIndex();

        // Start search a single index
        final DocIdAndDistance[] results = new DocIdAndDistance[searchParameters.getK()];
        for (int i = 0; i < searchParameters.getK(); i++) {
            results[i] = new DocIdAndDistance(DocIdAndDistance.INVALID_DOC_ID, 0);
        }

        try {
            faissIndex.searchLeaf(partialLoadingContext.getIndexInput(), results, searchParameters);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Transform results to query results
        int idx = results.length - 1;
        while (idx >= 0 && results[idx].id == DocIdAndDistance.INVALID_DOC_ID) {
            --idx;
        }
        // Ex: [id0, id2, id3, -1, -1, -1] where k == 6, then resultSize = 3
        final int resultSize = idx + 1;

        final KNNQueryResult[] queryResults = new KNNQueryResult[resultSize];
        for (int i = 0; i < resultSize; i++) {
            queryResults[i] = new KNNQueryResult(results[i].id, results[i].distance);
        }
        return queryResults;
    }
}
