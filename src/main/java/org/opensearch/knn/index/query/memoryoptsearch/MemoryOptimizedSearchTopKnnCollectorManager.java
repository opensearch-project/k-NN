/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.MultiLeafKnnCollector;
import org.apache.lucene.util.hnsw.BlockingFloatHeap;
import org.opensearch.common.CheckedTriFunction;

import java.io.IOException;

/**
 * KnnCollector manager by wrapping {@link AbstractKnnCollector} with {@link MultiLeafKnnCollector}
 * when multiple segments are present.
 * <p>
 * Internally, it maintains a global min-heap to track the lowest competitive similarity found so far.
 * This can accelerate convergence in each search thread when concurrent search is enabled.
 */
public class MemoryOptimizedSearchTopKnnCollectorManager implements KnnCollectorManager {

    // the number of docs to collect
    private final int k;
    // the global score queue used to track the top scores collected across all leaves
    private final BlockingFloatHeap globalScoreQueue;
    // KnnCollector supplier.
    private final CheckedTriFunction<Integer, KnnSearchStrategy, LeafReaderContext, AbstractKnnCollector, IOException> knnCollectorSupplier;

    /**
     * @param k The number of documents to collect.
     * @param indexSearcher IndexSearcher having LeafReaders underlying.
     * @param knnCollectorSupplier KnnCollector supplier, which must not return null.
     */
    public MemoryOptimizedSearchTopKnnCollectorManager(
        final int k,
        final IndexSearcher indexSearcher,
        final CheckedTriFunction<Integer, KnnSearchStrategy, LeafReaderContext, AbstractKnnCollector, IOException> knnCollectorSupplier
    ) {
        final boolean isMultiSegments = indexSearcher.getIndexReader().leaves().size() > 1;
        this.k = k;
        this.globalScoreQueue = isMultiSegments ? new BlockingFloatHeap(k) : null;
        this.knnCollectorSupplier = knnCollectorSupplier;
    }

    /**
     * Return a new {@link KnnCollector} instance.
     *
     * @param visitedLimit the maximum number of nodes that the search is allowed to visit
     * @param context the leaf reader context
     */
    @Override
    public KnnCollector newCollector(final int visitedLimit, final KnnSearchStrategy searchStrategy, final LeafReaderContext context) {
        try {
            if (globalScoreQueue == null) {
                return knnCollectorSupplier.apply(visitedLimit, searchStrategy, context);
            } else {
                return new MultiLeafKnnCollector(k, globalScoreQueue, knnCollectorSupplier.apply(visitedLimit, searchStrategy, context));
            }
        } catch (final IOException e) {
            throw new RuntimeException(e);
        }
    }
}
