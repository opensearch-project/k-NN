/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.DiversifyingNearestChildrenKnnCollectorManager;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.MemoryOptimizedSearchScoreConverter;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.lucene.OptimisticKnnCollectorManager;
import org.opensearch.lucene.ReentrantKnnCollectorManager;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO;
import static org.opensearch.knn.plugin.stats.KNNCounter.GRAPH_QUERY_ERRORS;

/**
 * Calculates query weights and builds query scorers.
 * Internally, it relies on memory-optimized search logic to perform vector search,
 * which sets it apart from {@link org.opensearch.knn.index.query.DefaultKNNWeight},
 * where the entire data is loaded into off-heap memory.
 */
@Log4j2
public class MemoryOptimizedKNNWeight extends KNNWeight {
    // ACORN activates when the filtering rate is below this threshold, so 0 ensures no filtering rate
    // can satisfy it, effectively disabling ACORN. This matches Lucene 10.4's decision to disable ACORN
    // due to observed recall degradation with filtered searches (see: https://github.com/opensearch-project/k-NN/issues/3327).
    private static final KnnSearchStrategy.Hnsw DEFAULT_HNSW_SEARCH_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    private final KnnCollectorManager knnCollectorManager;
    private final IndexSearcher searcher;
    @Setter
    private ReentrantKnnCollectorManager reentrantKNNCollectorManager;

    public MemoryOptimizedKNNWeight(KNNQuery query, float boost, final Weight filterWeight, IndexSearcher searcher, Integer k) {
        super(query, boost, filterWeight);
        this.searcher = searcher;

        if (k != null && k > 0) {
            // ANN Search
            if (query.getParentsFilter() == null) {
                // Non-nested case
                this.knnCollectorManager = new OptimisticKnnCollectorManager(k, new TopKnnCollectorManager(k, searcher));
            } else {
                // Nested case
                this.knnCollectorManager = new DiversifyingNearestChildrenKnnCollectorManager(k, query.getParentsFilter(), searcher);
            }
        } else {
            // Radius search.
            // Forward the incoming searchStrategy so that a re-entrant (seeded) radial search can begin
            // the graph traversal from pre-computed entry points instead of the default entry node.
            this.knnCollectorManager = (visitLimit, searchStrategy, context) -> new RadiusVectorSimilarityCollector(
                DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * query.getRadius(),
                query.getRadius(),
                visitLimit,
                searchStrategy
            );
        }
    }

    @Override
    protected TopDocs doANNSearch(
        final LeafReaderContext context,
        final SegmentReader reader,
        final FieldInfo fieldInfo,
        final SpaceType spaceType,
        final KNNEngine knnEngine,
        final VectorDataType vectorDataType,
        final byte[] quantizedTargetVector,
        final float[] adcTransformedVector,
        final String modelId,
        final BitSet filterIdsBitSet,
        final int cardinality,
        final int k
    ) {
        try {
            if (k > 0) {
                // KNN search
                if (quantizedTargetVector != null) {
                    // Quantization case
                    if (quantizationService.getVectorDataTypeForTransfer(fieldInfo) == VectorDataType.BINARY) {
                        return queryIndex(
                            quantizedTargetVector,
                            cardinality,
                            cardinality + 1,
                            context,
                            filterIdsBitSet,
                            reader,
                            knnEngine,
                            spaceType
                        );
                    }

                    // Should never occur, safety if ever any other quantization is added
                    throw new IllegalStateException(
                        "VectorDataType for transfer acquired ["
                            + quantizationService.getVectorDataTypeForTransfer(fieldInfo)
                            + "] while it is expected to get ["
                            + VectorDataType.BINARY
                            + "]"
                    );
                }

                if (knnQuery.getVectorDataType() == VectorDataType.BINARY || knnQuery.getVectorDataType() == VectorDataType.BYTE) {
                    // when data_type is set byte or binary
                    return queryIndex(
                        knnQuery.getByteQueryVector(),
                        cardinality,
                        cardinality + 1,
                        context,
                        filterIdsBitSet,
                        reader,
                        knnEngine,
                        spaceType
                    );
                }

                if (adcTransformedVector != null) {
                    // ADC case
                    return queryIndex(
                        adcTransformedVector,
                        cardinality,
                        cardinality + 1,
                        context,
                        filterIdsBitSet,
                        reader,
                        knnEngine,
                        spaceType
                    );
                }

                // fallback to float
                return queryIndex(
                    knnQuery.getQueryVector(),
                    cardinality,
                    cardinality + 1,
                    context,
                    filterIdsBitSet,
                    reader,
                    knnEngine,
                    spaceType
                );
            } else {
                // Radius (radial) search via the memory-optimized path.
                //
                // Instead of running Lucene's radial graph traversal from the default entry node, we:
                //   1. Run a top-k ANN search with k = ef_search to discover good entry points.
                //   2. Run the radial (similarity-threshold) search re-entrant from those seeds.
                return radialSearch(context, reader, knnEngine, spaceType, filterIdsBitSet, cardinality);
            }
        } catch (Exception e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }
    }

    /**
     * Two-phase radial search for the memory-optimized path.
     * <p>
     * Phase 1 performs a top-k approximate search with {@code k = ef_search} to collect a set of
     * high-quality entry points. Phase 2 runs the radial (similarity-threshold) search seeded with
     * those entry points via {@link ReentrantKnnCollectorManager}, so the graph traversal re-enters
     * from known-good nodes rather than the default graph entry node. If phase 1 yields no seeds, the
     * search falls back to a plain (un-seeded) radial search.
     */
    private TopDocs radialSearch(
        final LeafReaderContext context,
        final SegmentReader reader,
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final BitSet filterIdsBitSet,
        final int cardinality
    ) throws IOException {
        final Object targetVector = knnQuery.getVector();
        final int visitedLimit = getFilterWeight() == null ? Integer.MAX_VALUE : cardinality;
        final AcceptDocs acceptDocs = getAcceptedDocs(reader, cardinality, filterIdsBitSet);

        // Phase 1: top-k ANN with k = ef_search to find seed entry points.
        final int efSearch = IndexHyperParametersUtil.getHNSWEFSearchValue(knnQuery.getMethodParameters(), knnQuery.getIndexName());
        final KnnCollector seedCollector = new TopKnnCollectorManager(efSearch, searcher).newCollector(
            visitedLimit,
            DEFAULT_HNSW_SEARCH_STRATEGY,
            context
        );
        searchVector(targetVector, seedCollector, acceptDocs, reader);
        final TopDocs seedTopDocs = seedCollector.topDocs();

        // Phase 2: radial search, seeded from phase-1 results when available.
        final KnnCollectorManager radialCollectorManager;
        if (seedTopDocs != null && seedTopDocs.scoreDocs.length > 0) {
            radialCollectorManager = new ReentrantKnnCollectorManager(
                knnCollectorManager,
                Map.of(context.ord, seedTopDocs),
                targetVector,
                knnQuery.getField()
            );
        } else {
            radialCollectorManager = knnCollectorManager;
        }

        return queryIndex(targetVector, cardinality, cardinality, context, filterIdsBitSet, reader, knnEngine, spaceType, radialCollectorManager);
    }

    /**
     * Issues the vector search against the segment using the given collector.
     */
    private void searchVector(final Object targetVector, final KnnCollector collector, final AcceptDocs acceptDocs, final SegmentReader reader)
        throws IOException {
        assert (targetVector instanceof float[] || targetVector instanceof byte[]);
        if (targetVector instanceof float[] floatTargetVector) {
            reader.getVectorReader().search(knnQuery.getField(), floatTargetVector, collector, acceptDocs);
        } else {
            reader.getVectorReader().search(knnQuery.getField(), (byte[]) targetVector, collector, acceptDocs);
        }
    }

    private TopDocs queryIndex(
        final Object targetVector,
        final int cardinality,
        final int visitLimitWhenFilterExists,
        final LeafReaderContext context,
        final BitSet filterIdsBitSet,
        final SegmentReader reader,
        final KNNEngine knnEngine,
        final SpaceType spaceType
    ) throws IOException {
        final KnnCollectorManager collectorManager = reentrantKNNCollectorManager != null
            ? reentrantKNNCollectorManager
            : knnCollectorManager;
        return queryIndex(
            targetVector,
            cardinality,
            visitLimitWhenFilterExists,
            context,
            filterIdsBitSet,
            reader,
            knnEngine,
            spaceType,
            collectorManager
        );
    }

    private TopDocs queryIndex(
        final Object targetVector,
        final int cardinality,
        final int visitLimitWhenFilterExists,
        final LeafReaderContext context,
        final BitSet filterIdsBitSet,
        final SegmentReader reader,
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final KnnCollectorManager collectorManager
    ) throws IOException {
        assert (targetVector instanceof float[] || targetVector instanceof byte[]);

        // Determine visit limit
        final int visitedLimit;
        if (getFilterWeight() == null) {
            visitedLimit = Integer.MAX_VALUE;
        } else {
            visitedLimit = visitLimitWhenFilterExists;
        }

        // Create a collector + bitset
        final KnnCollector knnCollector = collectorManager.newCollector(visitedLimit, DEFAULT_HNSW_SEARCH_STRATEGY, context);
        final AcceptDocs acceptDocs = getAcceptedDocs(reader, cardinality, filterIdsBitSet);

        // Start searching index
        searchVector(targetVector, knnCollector, acceptDocs, reader);

        // Make results to return
        TopDocs topDocs = knnCollector.topDocs();
        // Align `hitCount` logic with the non-memory-optimized path by setting it to the size of the result set.
        // Note: DefaultKNNWeight defines `hitCount` as the number of results returned per Lucene segment,
        // while Lucene’s implementation interprets it as the total number of vectors visited during search. We will
        // preserve the totalHits relation so that we know if we exhausted Lucene's search budget.
        topDocs = new TopDocs(new TotalHits(topDocs.scoreDocs.length, topDocs.totalHits.relation()), topDocs.scoreDocs);
        if (topDocs.scoreDocs.length == 0) {
            log.debug("[KNN] Query yielded 0 results");
            return EMPTY_TOPDOCS;
        }
        if (spaceType == SpaceType.COSINESIMIL) {
            MemoryOptimizedSearchScoreConverter.convertToCosineScore(topDocs.scoreDocs);
        }
        addExplainIfRequired(topDocs, knnEngine, spaceType);
        return topDocs;
    }

    private AcceptDocs getAcceptedDocs(SegmentReader reader, int cardinality, BitSet filterIdsBitSet) {
        final AcceptDocs acceptDocs;
        if (cardinality == 0) {
            // We may want to use liveDocs here rather than null, to ensure that deleted docs are not considered in k-NN search
            // But we are not doing this, because in that case it will break the current behavior of LOF which is similar to
            // normal faiss based search.
            acceptDocs = AcceptDocs.fromLiveDocs(null, reader.maxDoc());
        } else {
            acceptDocs = new AcceptDocs() {
                @Override
                public Bits bits() {
                    return filterIdsBitSet;
                }

                @Override
                public DocIdSetIterator iterator() {
                    return new BitSetIterator(filterIdsBitSet, cardinality);
                }

                @Override
                public int cost() {
                    return cardinality;
                }
            };
        }
        return acceptDocs;
    }
}
