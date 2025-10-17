/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.io.IOException;

/**
 * Collector manager responsible for constructing the appropriate KNN collector
 * depending on whether the optimistic search strategy is enabled.
 *
 * <p>This class wraps an underlying {@code KnnCollectorManager} delegate and
 * decides at runtime which collector implementation to use:
 *
 * <ul>
 *   <li>If the collector manager is configured to use <b>optimistic search</b>, it creates
 *       an optimistic collector, which performs a two-phase KNN search
 *       to optimize performance by reducing redundant segment searches.</li>
 *   <li>Otherwise, it falls back to the standard {@link KnnCollectorManager}
 *       provided by the delegate, preserving default Lucene KNN search behavior.</li>
 * </ul>
 *
 * <p>The optimistic search strategy operates in two phases:
 * <ol>
 *   <li>Phase 1 – Executes KNN searches independently per segment
 *       with adjusted {@code k} values based on segment size and merges the results.</li>
 *   <li>Phase 2 – Deep search: Re-runs searches only on segments that have
 *       promising results (based on a global score threshold) to refine recall efficiently.</li>
 * </ol>
 *
 * <p>Example usage:
 * <pre>{@code
 * KnnCollectorManager baseManager = new DefaultKnnCollectorManager(...);
 * OptimisticKnnCollectorManager manager =
 *     new OptimisticKnnCollectorManager(baseManager, useOptimisticSearch);
 * KnnCollector collector = manager.newCollector();
 * }</pre>
 *
 * Ported from <a href="https://github.com/apache/lucene/blob/8e8e37d9e94c290cf8d02e9f318e601baedf28bc/lucene/core/src/java/org/apache/lucene/search/AbstractKnnVectorQuery.java#L251">...</a>
 */
public class OptimisticKnnCollectorManager implements KnnCollectorManager {
    // Constant controlling the degree of additional result exploration done during
    // pro-rata search of segments.
    private static final int LAMBDA = 16;

    private final int k;
    private final KnnCollectorManager delegate;

    public OptimisticKnnCollectorManager(int k, KnnCollectorManager delegate) {
        this.k = k;
        this.delegate = delegate;
    }

    @Override
    public KnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context) throws IOException {
        // The delegate supports optimistic collection
        if (delegate.isOptimistic()) {
            @SuppressWarnings("resource")
            float leafProportion = context.reader().maxDoc() / (float) context.parent.reader().maxDoc();
            int perLeafTopK = perLeafTopKCalculation(k, leafProportion);
            // if we divided by zero above, leafProportion can be NaN and then this would be 0
            assert perLeafTopK > 0;
            return delegate.newOptimisticCollector(visitedLimit, searchStrategy, context, perLeafTopK);
        }
        // We don't support optimistic collection, so just do regular execution path
        return delegate.newCollector(visitedLimit, searchStrategy, context);
    }

    /**
     * Returns perLeafTopK, the expected number (K * leafProportion) of hits in a leaf with the given
     * proportion of the entire index, plus three standard deviations of a binomial distribution. Math
     * says there is a 95% probability that this segment's contribution to the global top K hits are
     * <= perLeafTopK.
     */
    private static int perLeafTopKCalculation(int k, float leafProportion) {
        return (int) Math.max(1, k * leafProportion + LAMBDA * Math.sqrt(k * leafProportion * (1 - leafProportion)));
    }
}
