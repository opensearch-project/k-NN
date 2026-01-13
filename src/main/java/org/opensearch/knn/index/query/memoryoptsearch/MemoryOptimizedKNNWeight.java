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
import org.opensearch.lucene.OptimisticKnnCollectorManager;
import org.opensearch.lucene.ReentrantKnnCollectorManager;

import java.io.IOException;

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
    // Enable ACORN optimization when having filtering rate < 60%.
    private static final KnnSearchStrategy.Hnsw DEFAULT_HNSW_SEARCH_STRATEGY = new KnnSearchStrategy.Hnsw(60);

    private final KnnCollectorManager knnCollectorManager;
    @Setter
    private ReentrantKnnCollectorManager reentrantKNNCollectorManager;

    public MemoryOptimizedKNNWeight(KNNQuery query, float boost, final Weight filterWeight, IndexSearcher searcher, Integer k) {
        super(query, boost, filterWeight);

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
            // Radius search
            this.knnCollectorManager = (visitLimit, searchStrategy, context) -> new RadiusVectorSimilarityCollector(
                DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * query.getRadius(),
                query.getRadius(),
                visitLimit
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
            // Get the appropriate search function based on field configuration
            final VectorSearchFunction searchFunction = SearchVectorTypeResolver.getSearchFunction(reader, fieldInfo, vectorDataType);

            if (k > 0) {
                // KNN search - determine which vector to use
                final Object targetVector;
                if (quantizedTargetVector != null) {
                    targetVector = quantizedTargetVector;
                } else if (adcTransformedVector != null) {
                    targetVector = adcTransformedVector;
                } else if (vectorDataType == VectorDataType.BINARY || vectorDataType == VectorDataType.BYTE) {
                    targetVector = knnQuery.getByteQueryVector();
                } else {
                    targetVector = knnQuery.getQueryVector();
                }

                return queryIndex(
                    targetVector,
                    searchFunction,
                    cardinality,
                    cardinality + 1,
                    context,
                    filterIdsBitSet,
                    reader,
                    knnEngine,
                    spaceType
                );
            } else {
                // Radius search
                return queryIndex(
                    knnQuery.getVector(),
                    searchFunction,
                    cardinality,
                    cardinality,
                    context,
                    filterIdsBitSet,
                    reader,
                    knnEngine,
                    spaceType
                );
            }
        } catch (Exception e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }
    }

    private TopDocs queryIndex(
        final Object targetVector,
        final VectorSearchFunction searchFunction,
        final int cardinality,
        final int visitLimitWhenFilterExists,
        final LeafReaderContext context,
        final BitSet filterIdsBitSet,
        final SegmentReader reader,
        final KNNEngine knnEngine,
        final SpaceType spaceType
    ) throws IOException {
        // Determine visit limit
        final int visitedLimit;
        if (getFilterWeight() == null) {
            visitedLimit = Integer.MAX_VALUE;
        } else {
            visitedLimit = visitLimitWhenFilterExists;
        }

        // Create a collector + bitset
        final KnnCollectorManager collectorManager = reentrantKNNCollectorManager != null
            ? reentrantKNNCollectorManager
            : knnCollectorManager;
        final KnnCollector knnCollector = collectorManager.newCollector(visitedLimit, DEFAULT_HNSW_SEARCH_STRATEGY, context);
        final AcceptDocs acceptDocs = getAcceptedDocs(reader, cardinality, filterIdsBitSet);

        // Start searching index using the provided search function
        searchFunction.search(knnQuery.getField(), targetVector, knnCollector, acceptDocs);

        // Make results to return
        TopDocs topDocs = knnCollector.topDocs();
        // Align `hitCount` logic with the non-memory-optimized path by setting it to the size of the result set.
        // Note: DefaultKNNWeight defines `hitCount` as the number of results returned per Lucene segment,
        // while Lucene's implementation interprets it as the total number of vectors visited during search.
        topDocs = new TopDocs(new TotalHits(topDocs.scoreDocs.length, TotalHits.Relation.EQUAL_TO), topDocs.scoreDocs);
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
