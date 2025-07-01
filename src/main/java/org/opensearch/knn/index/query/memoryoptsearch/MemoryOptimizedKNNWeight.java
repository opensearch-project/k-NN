/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.DiversifyingNearestChildrenKnnCollectorManager;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;

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
    private final KnnCollectorManager knnCollectorManager;

    public MemoryOptimizedKNNWeight(KNNQuery query, float boost, final Weight filterWeight, IndexSearcher searcher, int k) {
        super(query, boost, filterWeight);

        if (k > 0) {
            // ANN Search
            if (query.getParentsFilter() == null) {
                // Non-nested case
                this.knnCollectorManager = new TopKnnCollectorManager(k, searcher);
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
            final Version segmentLuceneVersion = reader.getSegmentInfo().info.getVersion();
            if (k > 0) {
                // KNN search
                if (quantizedTargetVector != null) {
                    // Quantization case
                    if (quantizationService.getVectorDataTypeForTransfer(fieldInfo, segmentLuceneVersion) == VectorDataType.BINARY) {
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
                            + quantizationService.getVectorDataTypeForTransfer(fieldInfo, segmentLuceneVersion)
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
                // Radius search
                return queryIndex(
                    knnQuery.getQueryVector(),
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
        final int cardinality,
        final int visitLimitWhenFilterExists,
        final LeafReaderContext context,
        final BitSet filterIdsBitSet,
        final SegmentReader reader,
        final KNNEngine knnEngine,
        final SpaceType spaceType
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
        final KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, KnnSearchStrategy.Hnsw.DEFAULT, context);
        final BitSet bitSet = cardinality == 0 ? null : filterIdsBitSet;

        // Start searching index
        if (targetVector instanceof float[] floatTargetVector) {
            reader.getVectorReader().search(knnQuery.getField(), floatTargetVector, knnCollector, bitSet);
        } else {
            reader.getVectorReader().search(knnQuery.getField(), (byte[]) targetVector, knnCollector, bitSet);
        }

        // Make results to return
        final TopDocs topDocs = knnCollector.topDocs();
        if (topDocs.scoreDocs.length == 0) {
            log.debug("[KNN] Query yielded 0 results");
            return EMPTY_TOPDOCS;
        }
        addExplainIfRequired(topDocs, knnEngine, spaceType);
        return topDocs;
    }
}
