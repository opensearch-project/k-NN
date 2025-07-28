/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;

/**
 * This is a Vector format that will be used for Native engines like Faiss and Nmslib for reading and writing vector
 * related data structures.
 */
@Log4j2
public class NativeEngines990KnnVectorsFormat extends KnnVectorsFormat {
    /** The format for storing, reading, merging vectors on disk */
    private static FlatVectorsFormat flatVectorsFormat;
    private static final String FORMAT_NAME = "NativeEngines990KnnVectorsFormat";
    private static int approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private final Optional<MapperService> mapperService;

    public NativeEngines990KnnVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public NativeEngines990KnnVectorsFormat(int approximateThreshold) {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()), approximateThreshold);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        this(flatVectorsFormat, KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat, int approximateThreshold) {
        this(flatVectorsFormat, approximateThreshold, new NativeIndexBuildStrategyFactory(), Optional.empty());
    }

    public NativeEngines990KnnVectorsFormat(
        final FlatVectorsFormat flatVectorsFormat,
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        final Optional<MapperService> mapperService
    ) {
        super(FORMAT_NAME);
        NativeEngines990KnnVectorsFormat.flatVectorsFormat = flatVectorsFormat;
        NativeEngines990KnnVectorsFormat.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
        this.mapperService = mapperService;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        if (mapperService.isPresent()) {
            SegmentInfo info = state.segmentInfo;
            String indexName = mapperService.get().index().getName();
            info.putAttribute("index_name", indexName);
            info.putAttribute("warmup_enabled", String.valueOf(KNNSettings.isKnnIndexWarmupEnabled(indexName)));
        }

        return new NativeEngines990KnnVectorsWriter(
            state,
            flatVectorsFormat.fieldsWriter(state),
            approximateThreshold,
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        String indexName = state.segmentInfo.getAttribute("index_name");
        String warmupEnabled = state.segmentInfo.getAttribute("warmup_enabled");
        if (indexName != null && warmupEnabled.equals("true")) {
            for (final FieldInfo fieldInfo : state.fieldInfos) {
                final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, state.segmentInfo);
                if (vectorIndexFileName == null) {
                    continue;
                }
                final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, state.segmentInfo);
                final NativeMemoryCacheManager cacheManager = NativeMemoryCacheManager.getInstance();
                try {
                    final String spaceTypeName = fieldInfo.attributes().getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
                    final SpaceType spaceType = SpaceType.getSpace(spaceTypeName);
                    final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
                    final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
                    final QuantizationParams quantizationParams = QuantizationService.getInstance()
                        .getQuantizationParams(fieldInfo, state.segmentInfo.getVersion());
                    cacheManager.get(
                        new NativeMemoryEntryContext.IndexEntryContext(
                            state.segmentInfo.dir,
                            cacheKey,
                            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                            getParametersAtLoading(spaceType, knnEngine, indexName, vectorDataType, quantizationParams),
                            indexName
                        ),
                        true
                    );
                } catch (ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    /**
     * @param s
     * @return
     */
    @Override
    public int getMaxDimensions(String s) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", flatVectorsFormat="
            + flatVectorsFormat
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}
