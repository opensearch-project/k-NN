/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.cagra;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;

import java.io.IOException;

/**
 * This represents CAGRA HNSW.
 * Its logical structure is the same as the one built on a CPU.
 * The only difference are entry point and the number of layers, as searches in CAGRA HNSW are always conducted on the bottom layer.
 */
public class FaissCagraHNSW extends FaissHNSW {
    @Setter
    @Getter
    private int numBaseLevelSearchEntryPoints;

    /**
     * Partial load the CAGRA HNSW graph.
     * During a search on CAGRA HNSW, the entry point is randomly selected, and the search runs on the bottom graph.
     * As a result, the `entryPoint` and `maxLevel` variables remain at their default values. However, we need to initialize them correctly
     * to integrate with Lucene's HNSW graph searcher for proper functionality.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param totalNumberOfVectors The total number of vectors stored in the graph.
     * <p>
     * See : <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.cpp#L956">...</a>
     * @throws IOException
     */
    public void load(IndexInput input, long totalNumberOfVectors) throws IOException {
        super.load(input, totalNumberOfVectors);

        // Entry points will be provided via `RandomEntryPointsKnnSearchStrategy`, assigning -1 here.
        entryPoint = -1;

        // Result graph has 1-layer, but this info is not saved in file (having -1), so we override it in here.
        maxLevel = 1;
    }
}
