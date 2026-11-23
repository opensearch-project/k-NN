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
 * Its logical structure is the same as the one built on a CPU, and it comes in two shapes in FAISS depending on `base_level_only`:
 * <ul>
 *   <li>Base level only: the graph only has the bottom layer, having neither a valid entry point nor upper levels. Searches are
 *   conducted on the bottom layer, starting from randomly selected entry points.</li>
 *   <li>Multi-layer: upper levels were built using CPU on top of the GPU-built CAGRA bottom layer, so the graph has a valid entry point
 *   and max level. It is searched like a regular HNSW graph, descending from the entry point through the upper levels.</li>
 * </ul>
 */
public class FaissCagraHNSW extends FaissHNSW {
    @Setter
    @Getter
    private int numBaseLevelSearchEntryPoints;

    // True when searches must be conducted on the bottom layer only with randomly selected entry points.
    // False when the graph has a usable hierarchy, and therefore can be searched as a regular HNSW graph.
    @Getter
    private boolean baseLevelOnly = true;

    /**
     * Partial load the CAGRA HNSW graph, defaulting as a base-level-only graph.
     * See {@link #load(IndexInput, long, boolean)}.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param totalNumberOfVectors The total number of vectors stored in the graph.
     * @throws IOException
     */
    @Override
    public void load(IndexInput input, long totalNumberOfVectors) throws IOException {
        load(input, totalNumberOfVectors, true);
    }

    /**
     * Partial load the CAGRA HNSW graph.
     * <p>
     * When the graph only has the base level, the entry point is randomly selected during a search, and the search runs on the bottom
     * graph. As a result, the `entryPoint` and `maxLevel` variables remain at their default values (-1) in the file. However, we need
     * to initialize them correctly to integrate with Lucene's HNSW graph searcher for proper functionality.
     * <p>
     * When the graph has upper levels (e.g. `base_level_only=false`), keep the `entryPoint` and `maxLevel` saved in the file as they
     * are so that Lucene's HNSW graph searcher can descend from the entry point through the upper levels. If the file says it has
     * upper levels but does not have a usable hierarchy (i.e. no valid entry point or no upper level), fall back to base-level-only.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param totalNumberOfVectors The total number of vectors stored in the graph.
     * @param baseLevelOnlyFlag The `base_level_only` flag saved in a FAISS CAGRA index file.
     * <p>
     * See : <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.cpp#L956">...</a>
     * @throws IOException
     */
    public void load(IndexInput input, long totalNumberOfVectors, boolean baseLevelOnlyFlag) throws IOException {
        super.load(input, totalNumberOfVectors);

        // Validate that files flagged as multi-layer actually contain a valid entry point and at least one upper level for search
        // as a regular HNSW graph. Otherwise, fall back to searching the bottom layer with random entry points.
        final boolean isMultiLayer = !baseLevelOnlyFlag && entryPoint >= 0 && maxLevel >= 1;
        this.baseLevelOnly = !isMultiLayer;

        if (this.baseLevelOnly) {
            // Entry points will be provided via `RandomEntryPointsKnnSearchStrategy`.
            entryPoint = -1;

            // maxLevel is the 0-based index of the top level; 0 indicates that it only has the bottom layer.
            maxLevel = 0;
        }
    }
}
