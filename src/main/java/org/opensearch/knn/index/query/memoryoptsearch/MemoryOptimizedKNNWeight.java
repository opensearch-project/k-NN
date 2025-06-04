/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Collectors;

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
                this.knnCollectorManager = new MemoryOptimizedSearchTopKnnCollectorManager(
                    k,
                    searcher,
                    (visitLimit, searchStrategy, context) -> {
                        final int[] parentIds = getParentIdsArray(context);
                        return new GroupedTopKnnCollector(k, visitLimit, searchStrategy, BitSetParentIdGrouper.createGrouper(parentIds));
                    }
                );
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
    protected Map<Integer, Float> doANNSearch(
        String vectorIndexFileName,
        LeafReaderContext context,
        SegmentReader reader,
        FieldInfo fieldInfo,
        SpaceType spaceType,
        KNNEngine knnEngine,
        VectorDataType vectorDataType,
        byte[] quantizedTargetVector,
        String modelId,
        BitSet filterIdsBitSet,
        int cardinality,
        int k
    ) {
        KNNCounter.GRAPH_QUERY_REQUESTS.increment();

        try {
            if (k > 0) {
                // KNN search
                final TargetVector targetVector;
                if (knnQuery.getVectorDataType() == VectorDataType.BINARY
                    || (quantizedTargetVector != null
                        && quantizationService.getVectorDataTypeForTransfer(fieldInfo) == VectorDataType.BINARY)) {
                    final byte[] target = quantizedTargetVector == null ? knnQuery.getByteQueryVector() : quantizedTargetVector;
                    targetVector = new TargetVector(target);
                } else {
                    if (knnQuery.getVectorDataType() == VectorDataType.BYTE) {
                        targetVector = new TargetVector(knnQuery.getByteQueryVector());
                    } else {
                        targetVector = new TargetVector(knnQuery.getQueryVector());
                    }
                }

                return queryIndex(targetVector, cardinality, cardinality + 1, context, filterIdsBitSet, reader, knnEngine, spaceType);
            } else {
                final TargetVector targetVector;
                if (knnQuery.getVectorDataType() == VectorDataType.BYTE) {
                    targetVector = new TargetVector(knnQuery.getByteQueryVector());
                } else {
                    targetVector = new TargetVector(knnQuery.getQueryVector());
                }

                // Radius search
                return queryIndex(targetVector, cardinality, cardinality, context, filterIdsBitSet, reader, knnEngine, spaceType);
            }
        } catch (Exception e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }
    }

    private Map<Integer, Float> queryIndex(
        final TargetVector targetVector,
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
        final KnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, KnnSearchStrategy.Hnsw.DEFAULT, context);
        final BitSet bitSet = cardinality == 0 ? null : filterIdsBitSet;

        // Start searching index
        if (targetVector.isFloat()) {
            reader.getVectorReader().search(knnQuery.getField(), targetVector.asFloatVector(), knnCollector, bitSet);
        } else {
            reader.getVectorReader().search(knnQuery.getField(), targetVector.asByteVector(), knnCollector, bitSet);
        }

        // Make results to return
        final TopDocs topDocs = knnCollector.topDocs();
        return makeResults(topDocs, knnEngine, spaceType);
    }

    private Map<Integer, Float> makeResults(final TopDocs topDocs, KNNEngine knnEngine, SpaceType spaceType) {
        if (topDocs != null && topDocs.scoreDocs != null && topDocs.scoreDocs.length > 0) {
            // Add explanations if required, then return results
            final KNNQueryResult[] results = new KNNQueryResult[topDocs.scoreDocs.length];
            int i = 0;
            for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
                results[i] = new KNNQueryResult(scoreDoc.doc, scoreDoc.score);
                ++i;
            }

            addExplainIfRequired(results, knnEngine, spaceType);

            return Arrays.stream(results).collect(Collectors.toMap(KNNQueryResult::getId, KNNQueryResult::getScore));
        }

        log.debug("[KNN] Query yielded 0 results");
        return Collections.emptyMap();
    }

    private static class TargetVector {
        private final Object target;
        @Getter
        private final boolean isFloat;

        public TargetVector(final byte[] byteTarget) {
            this.target = byteTarget;
            this.isFloat = false;
        }

        public TargetVector(final float[] floatTarget) {
            this.target = floatTarget;
            this.isFloat = true;
        }

        public byte[] asByteVector() {
            assert (isFloat == false);
            return (byte[]) target;
        }

        public float[] asFloatVector() {
            assert (isFloat);
            return (float[]) target;
        }
    }
}
