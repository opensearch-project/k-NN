/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.SegmentProfilerUtil;
import org.opensearch.knn.plugin.transport.KNNIndexShardProfileResult;
import org.opensearch.knn.profiler.SegmentProfilerState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFilePrefix;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileSuffix;

/**
 * KNNIndexShard wraps IndexShard and adds methods to perform k-NN related operations against the shard
 */
@Log4j2
public class KNNIndexShard {
    private final IndexShard indexShard;
    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private static final String INDEX_SHARD_CLEAR_CACHE_SEARCHER = "knn-clear-cache";

    /**
     * Constructor to generate KNNIndexShard. We do not perform validation that the index the shard is from
     * is in fact a k-NN Index (index.knn = true). This may make sense to add later, but for now the operations for
     * KNNIndexShards that are not from a k-NN index should be no-ops.
     *
     * @param indexShard IndexShard to be wrapped.
     */
    public KNNIndexShard(IndexShard indexShard) {
        this.indexShard = indexShard;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
    }

    /**
     * Return the underlying IndexShard
     *
     * @return IndexShard
     */
    public IndexShard getIndexShard() {
        return indexShard;
    }

    /**
     * Return the name of the shards index
     *
     * @return Name of shard's index
     */
    public String getIndexName() {
        return indexShard.shardId().getIndexName();
    }

    public KNNIndexShardProfileResult profile(final String field) {
        try (Engine.Searcher searcher = indexShard.acquireSearcher("knn-profile")) {
            log.info("[KNN] Profiling field [{}] in index [{}]", field, getIndexName());

            List<SegmentProfilerState> segmentLevelProfilerStates = new ArrayList<>();

            // For each leaf, collect the profile
            for (LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
                try {
                    SegmentProfilerState state = SegmentProfilerUtil.getSegmentProfileState(leaf.reader(), field);
                    segmentLevelProfilerStates.add(state);
                    log.debug("[KNN] Successfully profiled segment for field [{}]", field);
                } catch (IOException e) {
                    log.error("[KNN] Failed to profile segment for field [{}]: {}", field, e.getMessage(), e);
                    throw new RuntimeException("Failed to profile segment: " + e.getMessage(), e);
                }
            }

            if (segmentLevelProfilerStates.isEmpty()) {
                log.warn("[KNN] No segments found with field [{}] in index [{}]", field, getIndexName());
            } else {
                log.info(
                    "[KNN] Successfully profiled [{}] segments for field [{}] in index [{}]",
                    segmentLevelProfilerStates.size(),
                    field,
                    getIndexName()
                );
            }

            return new KNNIndexShardProfileResult(segmentLevelProfilerStates, indexShard.shardId().toString());
        } catch (Exception e) {
            log.error("[KNN] Error profiling field [{}] in index [{}]: {}", field, getIndexName(), e.getMessage(), e);
            throw new RuntimeException("Error during profiling: " + e.getMessage(), e);
        }
    }

    /**
     * Load all of the k-NN segments for this shard into the cache.
     *
     * @throws IOException Thrown when getting the HNSW Paths to be loaded in
     */
    public void warmup() throws IOException {
        log.info("[KNN] Warming up index: [{}]", getIndexName());
        final Directory directory = indexShard.store().directory();
        try (Engine.Searcher searcher = indexShard.acquireSearcher("knn-warmup")) {
            getAllEngineFileContexts(searcher.getIndexReader()).forEach((engineFileContext) -> {
                try {
                    final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(
                        engineFileContext.vectorFileName,
                        engineFileContext.segmentInfo
                    );
                    nativeMemoryCacheManager.get(
                        new NativeMemoryEntryContext.IndexEntryContext(
                            directory,
                            cacheKey,
                            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                            getParametersAtLoading(
                                engineFileContext.getSpaceType(),
                                KNNEngine.getEngineNameFromPath(engineFileContext.getVectorFileName()),
                                getIndexName(),
                                engineFileContext.getVectorDataType()
                            ),
                            getIndexName(),
                            engineFileContext.getModelId()
                        ),
                        true
                    );
                } catch (ExecutionException ex) {
                    throw new RuntimeException(ex);
                }
            });
        }
    }

    /**
     * Removes all the k-NN segments for this shard from the cache.
     * Adding write lock onto the NativeMemoryAllocation of the index that needs to be evicted from cache.
     * Write lock will be unlocked after the index is evicted. This locking mechanism is used to avoid
     * conflicts with queries fired on this index when the index is being evicted from cache.
     */
    public void clearCache() {
        String indexName = getIndexName();
        Optional<NativeMemoryAllocation> indexAllocationOptional;
        NativeMemoryAllocation indexAllocation;
        indexAllocationOptional = nativeMemoryCacheManager.getIndexMemoryAllocation(indexName);
        if (indexAllocationOptional.isPresent()) {
            indexAllocation = indexAllocationOptional.get();
            indexAllocation.writeLock();
            log.info("[KNN] Evicting index from cache: [{}]", indexName);
            try (Engine.Searcher searcher = indexShard.acquireSearcher(INDEX_SHARD_CLEAR_CACHE_SEARCHER)) {
                getAllEngineFileContexts(searcher.getIndexReader()).forEach((engineFileContext) -> {
                    final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(
                        engineFileContext.vectorFileName,
                        engineFileContext.segmentInfo
                    );
                    nativeMemoryCacheManager.invalidate(cacheKey);
                });
            } catch (IOException ex) {
                log.error("[KNN] Failed to evict index from cache: [{}]", indexName, ex);
                throw new RuntimeException(ex);
            } finally {
                indexAllocation.writeUnlock();
            }
        }
    }

    /**
     * For the given shard, get all of its engine file context objects
     *
     * @param indexReader IndexReader to read the information for each segment in the shard
     * @return List of engine contexts
     * @throws IOException Thrown when the SegmentReader is attempting to read the segments files
     */
    @VisibleForTesting
    List<EngineFileContext> getAllEngineFileContexts(IndexReader indexReader) throws IOException {
        List<EngineFileContext> engineFiles = new ArrayList<>();
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            engineFiles.addAll(getEngineFileContexts(indexReader, knnEngine));
        }
        return engineFiles;
    }

    List<EngineFileContext> getEngineFileContexts(IndexReader indexReader, KNNEngine knnEngine) throws IOException {
        List<EngineFileContext> engineFiles = new ArrayList<>();

        for (LeafReaderContext leafReaderContext : indexReader.leaves()) {
            SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
            String fileExtension = reader.getSegmentInfo().info.getUseCompoundFile()
                ? knnEngine.getCompoundExtension()
                : knnEngine.getExtension();

            for (FieldInfo fieldInfo : reader.getFieldInfos()) {
                if (fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
                    // Space Type will not be present on ES versions 7.1 and 7.4 because the only available space type
                    // was L2. So, if Space Type is not present, just fall back to L2
                    String spaceTypeName = fieldInfo.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
                    SpaceType spaceType = SpaceType.getSpace(spaceTypeName);
                    String modelId = fieldInfo.attributes().getOrDefault(MODEL_ID, null);
                    engineFiles.addAll(
                        getEngineFileContexts(
                            reader.getSegmentInfo(),
                            fieldInfo.name,
                            fileExtension,
                            spaceType,
                            modelId,
                            FieldInfoExtractor.extractQuantizationConfig(fieldInfo) == QuantizationConfig.EMPTY
                                ? VectorDataType.get(
                                    fieldInfo.attributes().getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
                                )
                                : VectorDataType.BINARY
                        )
                    );
                }
            }
        }
        return engineFiles;
    }

    @VisibleForTesting
    List<EngineFileContext> getEngineFileContexts(
        SegmentCommitInfo segmentCommitInfo,
        String fieldName,
        String fileExtension,
        SpaceType spaceType,
        String modelId,
        VectorDataType vectorDataType
    ) throws IOException {
        // Ex: 0_
        final String prefix = buildEngineFilePrefix(segmentCommitInfo.info.name);
        // Ex: _my_field.faiss
        final String suffix = buildEngineFileSuffix(fieldName, fileExtension);
        return segmentCommitInfo.files()
            .stream()
            .filter(fileName -> fileName.startsWith(prefix))
            .filter(fileName -> fileName.endsWith(suffix))
            .map(vectorFileName -> new EngineFileContext(spaceType, modelId, vectorFileName, vectorDataType, segmentCommitInfo.info))
            .collect(Collectors.toList());
    }

    @AllArgsConstructor
    @Getter
    @VisibleForTesting
    static class EngineFileContext {
        private final SpaceType spaceType;
        private final String modelId;
        private final String vectorFileName;
        private final VectorDataType vectorDataType;
        private final SegmentInfo segmentInfo;
    }
}
