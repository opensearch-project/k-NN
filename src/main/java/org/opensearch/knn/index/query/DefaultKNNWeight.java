/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.apache.lucene.util.BitSet;

import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.plugin.stats.KNNCounter.GRAPH_QUERY_ERRORS;

/**
 * Calculates query weights and builds query scorers.
 * Internally, it interacts with native memory and relies heavily on a native library.
 */
@Log4j2
public class DefaultKNNWeight extends KNNWeight {
    private final NativeMemoryCacheManager nativeMemoryCacheManager;

    public DefaultKNNWeight(KNNQuery query, float boost, Weight filterWeight) {
        super(query, boost, filterWeight);
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
    }

    @Override
    protected TopDocs doANNSearch(
        final LeafReaderContext context,
        final SegmentReader reader,
        final FieldInfo fieldInfo,
        final SpaceType spaceType,
        final KNNEngine knnEngine,
        final VectorDataType vectorDataType,
        final byte[] quantizedVector,
        final float[] transformedVector,
        final String modelId,
        final BitSet filterIdsBitSet,
        final int cardinality,
        final int k
    ) throws IOException {
        final List<String> engineFiles = KNNCodecUtil.getEngineFiles(
            knnEngine.getExtension(),
            knnQuery.getField(),
            reader.getSegmentInfo().info
        );
        final String vectorIndexFileName = engineFiles.get(0);
        final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, reader.getSegmentInfo().info);

        final Version segmentLuceneVersion = reader.getSegmentInfo().info.getVersion();
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(
            reader,
            fieldInfo,
            knnQuery.getField(),
            segmentLuceneVersion
        );

        // We need to first get index allocation
        NativeMemoryAllocation indexAllocation;
        try {
            indexAllocation = nativeMemoryCacheManager.get(
                new NativeMemoryEntryContext.IndexEntryContext(
                    reader.directory(),
                    cacheKey,
                    NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                    getParametersAtLoading(
                        spaceType,
                        knnEngine,
                        knnQuery.getIndexName(),
                        // TODO: In the future, more vector data types will be supported with quantization
                        quantizedVector == null ? vectorDataType : VectorDataType.BINARY,
                        (segmentLevelQuantizationInfo == null) ? null : segmentLevelQuantizationInfo.getQuantizationParams()
                    ),
                    knnQuery.getIndexName(),
                    modelId
                ),
                true
            );
        } catch (ExecutionException e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }

        // From cardinality select different filterIds type
        FilterIdsSelector filterIdsSelector = FilterIdsSelector.getFilterIdSelector(filterIdsBitSet, cardinality);
        long[] filterIds = filterIdsSelector.getFilterIds();
        FilterIdsSelector.FilterIdsSelectorType filterType = filterIdsSelector.getFilterType();
        // Now that we have the allocation, we need to readLock it
        indexAllocation.readLock();
        try {
            indexAllocation.incRef();
        } catch (IllegalStateException e) {
            indexAllocation.readUnlock();
            log.error("[KNN] Exception when allocation getting evicted: ", e);
            throw new RuntimeException("Failed to do kNN search when vector data structures getting evicted ", e);
        }
        KNNQueryResult[] results;
        try {
            if (indexAllocation.isClosed()) {
                throw new RuntimeException("Index has already been closed");
            }
            final int[] parentIds = getParentIdsArray(context);
            if (k > 0) {
                if (knnQuery.getVectorDataType() == VectorDataType.BINARY
                    || quantizedVector != null
                        && quantizationService.getVectorDataTypeForTransfer(fieldInfo, segmentLuceneVersion) == VectorDataType.BINARY) {
                    results = JNIService.queryBinaryIndex(
                        indexAllocation.getMemoryAddress(),
                        // TODO: In the future, quantizedVector can have other data types than byte
                        quantizedVector == null ? knnQuery.getByteQueryVector() : quantizedVector,
                        k,
                        knnQuery.getMethodParameters(),
                        knnEngine,
                        filterIds,
                        filterType.getValue(),
                        parentIds
                    );
                } else {
                    results = JNIService.queryIndex(
                        indexAllocation.getMemoryAddress(),
                        transformedVector == null ? knnQuery.getQueryVector() : transformedVector,
                        k,
                        knnQuery.getMethodParameters(),
                        knnEngine,
                        filterIds,
                        filterType.getValue(),
                        parentIds
                    );
                }
            } else {
                results = JNIService.radiusQueryIndex(
                    indexAllocation.getMemoryAddress(),
                    knnQuery.getQueryVector(),
                    knnQuery.getRadius(),
                    knnQuery.getMethodParameters(),
                    knnEngine,
                    knnQuery.getContext().getMaxResultWindow(),
                    filterIds,
                    filterType.getValue(),
                    parentIds
                );
            }
        } catch (Exception e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        } finally {
            indexAllocation.readUnlock();
            indexAllocation.decRef();
        }

        TopApproxKnnCollector collector = new TopApproxKnnCollector(
            k > 0 ? k : knnQuery.getContext().getMaxResultWindow(),
            knnEngine,
            quantizedVector != null ? SpaceType.HAMMING : spaceType
        );
        for (KNNQueryResult knnQueryResult : results) {
            collector.incVisitedCount(1);
            collector.collect(knnQueryResult.getId(), knnQueryResult.getScore());
        }
        TopDocs topDocs = collector.topDocs();
        addExplainIfRequired(results, knnEngine, spaceType);
        return topDocs;
    }
}
