/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.faiss.hnsw.FaissHNSW;
import org.opensearch.knn.partialloading.search.AbstractDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.DocIdAndDistance;
import org.opensearch.knn.partialloading.search.GroupedDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.PlainDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.distance.DistanceComputer;

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
                "Expected flat vector storage format under [" + IHNF + "] index type, but got " + faissIndex.getIndexType());
        }
        return faissHNSWFlatIndex;
    }

    @Override
    public void searchLeaf(IndexInput indexInput, DocIdAndDistance[] results, PartialLoadingSearchParameters searchParameters)
        throws IOException {
        // Determine result heap
        final AbstractDistanceMaxHeap resultMaxHeap;
        if (searchParameters.getDocIdGrouper() != null) {
            resultMaxHeap = new GroupedDistanceMaxHeap(searchParameters.getK(), searchParameters.getDocIdGrouper());
        } else {
            resultMaxHeap = new PlainDistanceMaxHeap(searchParameters.getK());
        }

        // Create distance computer
        final DistanceComputer distanceComputer =
            DistanceComputer.createDistanceFunctionFromFlatVector(storage.getCodes(), searchParameters);

        // Start HNSW searching
        hnsw.search(distanceComputer, resultMaxHeap, searchParameters, results);

        // Reverse (e.g. negation) if it needs to.
        if (DistanceComputer.needReverseScore(searchParameters.getSpaceType())) {
            for (final DocIdAndDistance result : results) {
                result.distance = -result.distance;
            }
        }
    }

    @Override
    public String getIndexType() {
        return IHNF;
    }
}
