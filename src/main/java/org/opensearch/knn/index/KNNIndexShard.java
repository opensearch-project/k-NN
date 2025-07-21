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
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFilePrefix;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileSuffix;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;

/**
 * KNNIndexShard wraps IndexShard and adds methods to perform k-NN related operations against the shard
 */
@Log4j2
public class KNNIndexShard {
    @Getter
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

    private Set<String> warmUpMemoryOptimizedSearcher(
        final LeafReader leafReader,
        final MapperService mapperService,
        final String indexName
    ) {

        final Set<FieldInfo> fieldsForMemoryOptimizedSearch = StreamSupport.stream(leafReader.getFieldInfos().spliterator(), false)
            .filter(fieldInfo -> fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD))
            .filter(fieldInfo -> {
                final MappedFieldType fieldType = mapperService.fieldType(fieldInfo.getName());

                if (fieldType instanceof KNNVectorFieldType knnFieldType) {
                    return MemoryOptimizedSearchSupportSpec.isSupportedFieldType(knnFieldType, indexName);
                }

                return false;
            })
            .collect(Collectors.toSet());

        final SegmentReader segmentReader = Lucene.segmentReader(leafReader);
        for (final FieldInfo field : fieldsForMemoryOptimizedSearch) {
            final String dataTypeStr = field.getAttribute(VECTOR_DATA_TYPE_FIELD);
            if (dataTypeStr == null) {
                continue;
            }
            try {
                // Partial load Faiss index by triggering search.
                final VectorDataType vectorDataType = VectorDataType.get(dataTypeStr);
                if (vectorDataType == VectorDataType.FLOAT) {
                    segmentReader.getVectorReader().search(field.getName(), (float[]) null, null, null);
                } else {
                    segmentReader.getVectorReader().search(field.getName(), (byte[]) null, null, null);
                }
            } catch (Exception e) {
                // Ignore
            }
        }

        return fieldsForMemoryOptimizedSearch.stream().map(FieldInfo::getName).collect(Collectors.toSet());
    }

    /**
     * Load all the k-NN segments for this shard into the cache.
     * First it tries to warm-up memory optimized fields, then load off-heap fields.
     *
     * @throws IOException Thrown when getting the HNSW Paths to be loaded in
     */
    public void warmup() throws IOException {
        log.info("[KNN] Warming up index: [{}]", getIndexName());

        final MapperService mapperService = indexShard.mapperService();
        final String indexName = indexShard.shardId().getIndexName();
        final Directory directory = indexShard.store().directory();

        try (Engine.Searcher searcher = indexShard.acquireSearcher("knn-warmup-mem")) {
            for (final LeafReaderContext leafReaderContext : searcher.getIndexReader().leaves()) {
                // Load memory optimized searcher in a single segment first.
                final Set<String> loadedFieldNames = warmUpMemoryOptimizedSearcher(leafReaderContext.reader(), mapperService, indexName);
                log.info("[KNN] Loaded memory optimized searchers for fields {}", loadedFieldNames);

                // Load off-heap index
                final List<EngineFileContext> engineFileContexts = getAllEngineFileContexts(loadedFieldNames, leafReaderContext);
                warmUpOffHeapIndex(engineFileContexts, directory);
                log.info("[KNN] Loaded off-heap indices for fields {}", engineFileContexts.stream().map(ctx -> ctx.fieldName));
            }
        }
    }

    private void warmUpOffHeapIndex(final List<EngineFileContext> engineFileContexts, final Directory directory) {
        for (final EngineFileContext engineFileContext : engineFileContexts) {
            try {
                // Get cache key for an off-heap index
                final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(
                    engineFileContext.vectorFileName,
                    engineFileContext.segmentInfo
                );

                // Load an off-heap index
                nativeMemoryCacheManager.get(
                    new NativeMemoryEntryContext.IndexEntryContext(
                        directory,
                        cacheKey,
                        NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                        getParametersAtLoading(
                            engineFileContext.getSpaceType(),
                            KNNEngine.getEngineNameFromPath(engineFileContext.getVectorFileName()),
                            getIndexName(),
                            engineFileContext.getVectorDataType(),
                            (engineFileContext.getSegmentLevelQuantizationInfo() == null) ? null : engineFileContext.getSegmentLevelQuantizationInfo().getQuantizationParams()
                        ),
                        getIndexName(),
                        engineFileContext.getModelId()
                    ),
                    true
                );
            } catch (ExecutionException ex) {
                throw new RuntimeException(ex);
            }
        }
    }

    /**
     * Removes all the k-NN segments for this shard from the cache.
     * Adding write lock onto the {@link NativeMemoryAllocation} of the index that needs to be evicted from cache.
     * Write lock will be unlocked after the index is evicted. This locking mechanism is used to avoid
     * conflicts with queries fired on this index when the index is being evicted from cache.
     */
    public void clearCache() {
        final String indexName = getIndexName();
        final Optional<NativeMemoryAllocation> indexAllocationOptional = nativeMemoryCacheManager.getIndexMemoryAllocation(indexName);
        if (indexAllocationOptional.isPresent()) {
            final NativeMemoryAllocation indexAllocation = indexAllocationOptional.get();
            indexAllocation.writeLock();
            log.info("[KNN] Evicting index from cache: [{}]", indexName);
            try (Engine.Searcher searcher = indexShard.acquireSearcher(INDEX_SHARD_CLEAR_CACHE_SEARCHER)) {
                for (final LeafReaderContext leafReaderContext : searcher.getIndexReader().leaves()) {
                    getAllEngineFileContexts(Collections.emptySet(), leafReaderContext).forEach(engineFileContext -> {
                        final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(
                            engineFileContext.vectorFileName,
                            engineFileContext.segmentInfo
                        );
                        nativeMemoryCacheManager.invalidate(cacheKey);
                    });
                }
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
     * @param loadedFieldNames Field names that are already warmed up do not need to load the off-heap index.
     * @param leafReaderContext LeafReader to read the information for each segment in the shard.
     * @return List of engine contexts
     * @throws IOException Thrown when the SegmentReader is attempting to read the segments files
     */
    @VisibleForTesting
    List<EngineFileContext> getAllEngineFileContexts(final Set<String> loadedFieldNames, final LeafReaderContext leafReaderContext)
        throws IOException {
        List<EngineFileContext> engineFiles = new ArrayList<>();
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            engineFiles.addAll(getEngineFileContexts(loadedFieldNames, leafReaderContext, knnEngine));
        }
        return engineFiles;
    }

    List<EngineFileContext> getEngineFileContexts(
        final Set<String> loadedFieldNames,
        final LeafReaderContext leafReaderContext,
        KNNEngine knnEngine
    ) throws IOException {
        final List<EngineFileContext> engineFiles = new ArrayList<>();
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final String fileExtension = reader.getSegmentInfo().info.getUseCompoundFile()
            ? knnEngine.getCompoundExtension()
            : knnEngine.getExtension();

        for (final FieldInfo fieldInfo : reader.getFieldInfos()) {
            if (loadedFieldNames.contains(fieldInfo.getName())) {
                continue;
            }

            if (fieldInfo.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)) {
                // Space Type will not be present on ES versions 7.1 and 7.4 because the only available space type
                // was L2. So, if Space Type is not present, just fall back to L2
                final String spaceTypeName = fieldInfo.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
                final SpaceType spaceType = SpaceType.getSpace(spaceTypeName);
                final String modelId = fieldInfo.attributes().getOrDefault(MODEL_ID, null);
                final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(
                    reader,
                    fieldInfo,
                    fieldInfo.name,
                    reader.getSegmentInfo().info.getVersion()
                );
                // obtain correct VectorDataType for this field based on the quantization state and if ADC is enabled.
                VectorDataType vectorDataType = determineVectorDataType(
                    fieldInfo,
                    segmentLevelQuantizationInfo,
                    reader.getSegmentInfo().info.getVersion()
                );

                engineFiles.addAll(
                    getEngineFileContexts(
                        reader.getSegmentInfo(),
                        segmentLevelQuantizationInfo,
                        fieldInfo.name,
                        fileExtension,
                        spaceType,
                        modelId,
                        vectorDataType
                    )
                );
            }
        }

        return engineFiles;
    }

    @VisibleForTesting
    List<EngineFileContext> getEngineFileContexts(
        final SegmentCommitInfo segmentCommitInfo,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        final String fieldName,
        final String fileExtension,
        final SpaceType spaceType,
        final String modelId,
        final VectorDataType vectorDataType
    ) throws IOException {
        // Ex: 0_
        final String prefix = buildEngineFilePrefix(segmentCommitInfo.info.name);
        // Ex: _my_field.faiss
        final String suffix = buildEngineFileSuffix(fieldName, fileExtension);
        return segmentCommitInfo.files()
            .stream()
            .filter(fileName -> fileName.startsWith(prefix))
            .filter(fileName -> fileName.endsWith(suffix))
            .map(
                vectorFileName -> new EngineFileContext(
                    fieldName,
                    spaceType,
                    modelId,
                    vectorFileName,
                    vectorDataType,
                    segmentCommitInfo.info,
                    segmentLevelQuantizationInfo
                )
            )
            .collect(Collectors.toList());
    }

    @VisibleForTesting
    VectorDataType determineVectorDataType(
        FieldInfo fieldInfo,
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        org.apache.lucene.util.Version segmentVersion
    ) {

        // First check if quantization config is empty
        if (FieldInfoExtractor.extractQuantizationConfig(fieldInfo, segmentVersion) == QuantizationConfig.EMPTY) {
            // If empty, get from attributes with default FLOAT
            return VectorDataType.get(fieldInfo.attributes().getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue()));
        }

        // For non-empty quantization config
        if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
            return VectorDataType.FLOAT;
        }

        return VectorDataType.BINARY;
    }

    @AllArgsConstructor
    @Getter
    @VisibleForTesting
    static class EngineFileContext {
        private final String fieldName;
        private final SpaceType spaceType;
        private final String modelId;
        private final String vectorFileName;
        private final VectorDataType vectorDataType;
        private final SegmentInfo segmentInfo;
        private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
    }
}
