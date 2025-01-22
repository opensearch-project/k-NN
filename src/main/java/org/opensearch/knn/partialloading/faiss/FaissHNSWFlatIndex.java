/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.partialloading.faiss.hnsw.FaissHNSW;
import org.opensearch.knn.partialloading.search.AbstractDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.GroupedDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.PlainDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.distance.DistanceComputer;

import java.io.IOException;

/**
 * A flat HNSW index that contains both an HNSW graph and flat vector storage.
 * This is the ported version of `IndexHNSW` from FAISS.
 * For more details, please refer to <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h">...</a>
 */
public class FaissHNSWFlatIndex extends FaissIndex {
    public static final String IHNF = "IHNf";

    private FaissHNSW hnsw = new FaissHNSW();
    private FaissIndexFlat storage;

    /**
     * Partially loads both the HNSW graph and the underlying flat vectors.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @return {@link FaissHNSWFlatIndex} instance consists of index hierarchy.
     * @throws IOException
     */
    public static FaissHNSWFlatIndex partiallyLoad(IndexInput input) throws IOException {
        // Read common header
        FaissHNSWFlatIndex faissHNSWFlatIndex = new FaissHNSWFlatIndex();
        readCommonHeader(input, faissHNSWFlatIndex);

        // Partial load HNSW graph
        faissHNSWFlatIndex.hnsw = FaissHNSW.partiallyLoad(input, faissHNSWFlatIndex.getTotalNumberOfVectors());

        // Partial load flat vector storage
        final FaissIndex faissIndex = FaissIndex.partiallyLoad(input);
        if (faissIndex instanceof FaissIndexFlat) {
            faissHNSWFlatIndex.storage = (FaissIndexFlat) faissIndex;
        } else {
            throw new IllegalStateException(
                "Expected flat vector storage format under [" + IHNF + "] index type, but got " + faissIndex.getIndexType());
        }
        return faissHNSWFlatIndex;
    }

    /**
     * Performs a KNN search on this index and updates the results with the nearest vectors found.
     *
     * @param indexInput An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param results A result array containing non-null pairs of vector IDs and their distances. After the search, it is updated by
     *                extracting elements from the result max-heap.
     * @param searchParameters HNSW search parameters, including efSearch, allow customization. If efSearch is provided, it will override
     *                        the default value.
     * @throws IOException
     */
    @Override
    public void search(IndexInput indexInput, IdAndDistance[] results, PartialLoadingSearchParameters searchParameters)
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

        // Reverse (e.g., negate) the distance if necessary.
        // This is required because internally, we rely on a max-heap based on distance to refine candidate vectors. For most metrics,
        // such as Euclidean distance, a greater distance means the vector is farther from the query vector. However, for some metrics,
        // for example inner product, a greater distance indicates the vector is closer to the query vector.
        // To make the distance work correctly with the max-heap, we negate the distance value before adding it to the heap.
        // For such metrics, we need to re-negate the distance afterward.
        if (DistanceComputer.needReverseScore(searchParameters.getSpaceType())) {
            for (final IdAndDistance result : results) {
                result.distance = -result.distance;
            }
        }
    }

    @Override
    public String getIndexType() {
        return IHNF;
    }
}
