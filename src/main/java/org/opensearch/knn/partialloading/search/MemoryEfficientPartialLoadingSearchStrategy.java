/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.partialloading.PartialLoadingContext;
import org.opensearch.knn.partialloading.faiss.FaissIndex;

import java.io.IOException;

public class MemoryEfficientPartialLoadingSearchStrategy extends PartialLoadingSearchStrategy {
    @Override
    public KNNQueryResult[] queryBinaryIndex(PartialLoadingSearchParameters searchParameters) {
        // TODO
        throw new UnsupportedOperationException("NOOOOOOOOOOOOOOOOOOOOOO");
    }

    @Override
    public KNNQueryResult[] queryIndex(PartialLoadingSearchParameters searchParameters) {
        // Prepare search
        PartialLoadingContext partialLoadingContext = searchParameters.getPartialLoadingContext();
        FaissIndex faissIndex = partialLoadingContext.getFaissIndex();

        // Start search a single index
        final DistanceMaxHeap resultMaxHeap = new DistanceMaxHeap(searchParameters.getK());
        final ResultsCollector resultsCollector = new HeapResultsCollector(resultMaxHeap);
        try {
            faissIndex.searchLeaf(partialLoadingContext.getIndexInput(), resultsCollector, searchParameters);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Translate distance to score
        final KNNQueryResult[] results = resultsCollector.getResults();
        final SpaceType spaceType = searchParameters.getSpaceType();
        for (int i = 0; i < results.length; ++i) {
            final KNNQueryResult result = results[i];
            result.reset(result.getId(), spaceType.scoreTranslation(result.getScore()));
        }
        return results;
    }
}
