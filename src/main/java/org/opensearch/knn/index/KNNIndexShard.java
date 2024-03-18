/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.FieldInfo;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFilePrefix;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileSuffix;

/**
 * KNNIndexShard wraps IndexShard and adds methods to perform k-NN related operations against the shard
 */
public class KNNIndexShard {
    private IndexShard indexShard;
    private NativeMemoryCacheManager nativeMemoryCacheManager;

    private static Logger logger = LogManager.getLogger(KNNIndexShard.class);

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

    /**
     * Load all of the k-NN segments for this shard into the cache.
     *
     * @throws IOException Thrown when getting the HNSW Paths to be loaded in
     */
    public void warmup() throws IOException {
        logger.info("[KNN] Warming up index: " + getIndexName());
        try (Engine.Searcher searcher = indexShard.acquireSearcher("knn-warmup")) {
            getAllEngineFileContexts(searcher.getIndexReader()).forEach((engineFileContext) -> {
                try {
                    nativeMemoryCacheManager.get(
                        new NativeMemoryEntryContext.IndexEntryContext(
                            engineFileContext.getIndexPath(),
                            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                            getParametersAtLoading(
                                engineFileContext.getSpaceType(),
                                KNNEngine.getEngineNameFromPath(engineFileContext.getIndexPath()),
                                getIndexName()
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
     * For the given shard, get all of its engine paths
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
            SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(leafReaderContext.reader());
            Path shardPath = ((FSDirectory) FilterDirectory.unwrap(reader.directory())).getDirectory();
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
                            reader.getSegmentInfo().files(),
                            reader.getSegmentInfo().info.name,
                            fieldInfo.name,
                            fileExtension,
                            shardPath,
                            spaceType,
                            modelId
                        )
                    );
                }
            }
        }
        return engineFiles;
    }

    @VisibleForTesting
    List<EngineFileContext> getEngineFileContexts(
        Collection<String> files,
        String segmentName,
        String fieldName,
        String fileExtension,
        Path shardPath,
        SpaceType spaceType,
        String modelId
    ) {
        String prefix = buildEngineFilePrefix(segmentName);
        String suffix = buildEngineFileSuffix(fieldName, fileExtension);
        return files.stream()
            .filter(fileName -> fileName.startsWith(prefix))
            .filter(fileName -> fileName.endsWith(suffix))
            .map(fileName -> shardPath.resolve(fileName).toString())
            .map(fileName -> new EngineFileContext(spaceType, modelId, fileName))
            .collect(Collectors.toList());
    }

    @AllArgsConstructor
    @Getter
    @VisibleForTesting
    static class EngineFileContext {
        private final SpaceType spaceType;
        private final String modelId;
        private final String indexPath;
    }
}
