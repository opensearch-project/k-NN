/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.partialloading.PartialLoadingContext;
import org.opensearch.knn.partialloading.faiss.FaissIndex;

import java.io.IOException;

/**
 * This strategy performs a vector search using {@link org.apache.lucene.store.IndexInput} to access bytes on-demand.
 * This approach minimizes the usage of JVM heap or native memory allocation.
 */
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
        final IdAndDistance[] results = new IdAndDistance[searchParameters.getK()];
        for (int i = 0; i < searchParameters.getK(); i++) {
            results[i] = new IdAndDistance(IdAndDistance.INVALID_DOC_ID, 0);
        }

        try {
            faissIndex.search(partialLoadingContext.getIndexInput(), results, searchParameters);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Get the actual returned result size. Even we gave `k` sized array, but vector search could end up finding `k`/2 results.
        int idx = results.length - 1;
        while (idx >= 0 && results[idx].id == IdAndDistance.INVALID_DOC_ID) {
            --idx;
        }
        // Ex: [id0, id2, id3, -1, -1, -1] where k == 6, then resultSize = 3
        final int resultSize = idx + 1;

        // Transform query results.
        final KNNQueryResult[] queryResults = new KNNQueryResult[resultSize];
        for (int i = 0; i < resultSize; i++) {
            queryResults[i] = new KNNQueryResult(results[i].id, results[i].distance);
        }
        return queryResults;
    }
}
