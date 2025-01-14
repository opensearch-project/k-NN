/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.faiss.hnsw.FaissHNSW;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.ResultsCollector;
import org.opensearch.knn.partialloading.util.DistanceComputer;

import java.io.IOException;

public class FaissHNSWFlatIndex extends FaissIndex {
    public static final String IHNF = "IHNf";

    private FaissHNSW hnsw = new FaissHNSW();
    private FaissIndexFlat storage;

    public static FaissIndex load(IndexInput input) throws IOException {
        FaissHNSWFlatIndex faissHNSWFlatIndex = new FaissHNSWFlatIndex();
        readCommonHeader(input, faissHNSWFlatIndex);
        faissHNSWFlatIndex.hnsw = FaissHNSW.readHnsw(input, faissHNSWFlatIndex.totalNumberOfVectors);
        final FaissIndex faissIndex = FaissIndex.load(input);
        if (faissIndex instanceof FaissIndexFlat) {
            faissHNSWFlatIndex.storage = (FaissIndexFlat) faissIndex;
        } else {
            throw new IllegalStateException(
                "Expected flat vector storage format under [" + IHNF + "] index type, but got " + faissIndex.getIndexType()
            );
        }
        return faissHNSWFlatIndex;
    }

    @Override
    public void searchLeaf(IndexInput indexInput, ResultsCollector resultsCollector, PartialLoadingSearchParameters searchParameters)
        throws IOException {
        // TODO : params->grp

        final DistanceComputer distanceComputer = DistanceComputer.createDistanceFunctionFromFlatVector(
            storage.getCodes(),
            searchParameters
        );
        hnsw.search(distanceComputer, resultsCollector, searchParameters);
    }

    @Override
    public String getIndexType() {
        return IHNF;
    }
}
