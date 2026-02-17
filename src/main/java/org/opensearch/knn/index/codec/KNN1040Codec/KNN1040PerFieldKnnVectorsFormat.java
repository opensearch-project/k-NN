/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120HnswBinaryVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.FaissCodecFormatResolver;
import org.opensearch.knn.index.engine.lucene.LuceneCodecFormatResolver;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Per-field KNN vectors format for the KNN1040 codec. Uses {@link Lucene99HnswVectorsFormat}
 * for HNSW, {@link Lucene99RWHnswScalarQuantizedVectorsFormat} for scalar quantization (to
 * preserve the {@code confidenceInterval} parameter), and
 * {@link Lucene104ScalarQuantizedVectorsFormat} with {@code SINGLE_BIT_QUERY_NIBBLE} encoding
 * for the flat BBQ method.
 */
public class KNN1040PerFieldKnnVectorsFormat extends KNN1040BasePerFieldKnnVectorsFormat {

    private static final Tuple<Integer, ExecutorService> DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE = Tuple.tuple(1, null);
    private static final ThreadPoolExecutor mergeExecutor;
    private static volatile int mergeThreadCount;

    static {
        // Use SynchronousQueue so maximumPoolSize governs actual thread creation (matching Lucene's
        // ConcurrentMergeScheduler.CachedExecutor pattern). CallerRunsPolicy provides graceful
        // degradation if a resize races with an in-flight merge submission.
        mergeExecutor = new ThreadPoolExecutor(
            0,
            1,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>(),
            new ThreadPoolExecutor.CallerRunsPolicy()
        );
        mergeExecutor.allowCoreThreadTimeOut(true);
    }

    /**
     * Called by KNNSettings when knn.algo_param.index_thread_qty changes at runtime.
     * Resizes the shared merge executor pool without creating a new instance.
     */
    public static void updateMergeThreadCount(int newCount) {
        mergeThreadCount = newCount;
        int targetCore = Math.max(newCount, 0);
        int targetMax = Math.max(newCount, 1);
        // Always adjust in the safe order to avoid core > max invariant violation
        if (targetMax >= mergeExecutor.getMaximumPoolSize()) {
            mergeExecutor.setMaximumPoolSize(targetMax);
            mergeExecutor.setCorePoolSize(targetCore);
        } else {
            mergeExecutor.setCorePoolSize(targetCore);
            mergeExecutor.setMaximumPoolSize(targetMax);
        }
    }

    public KNN1040PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        this(mapperService, new NativeIndexBuildStrategyFactory());
    }

    public KNN1040PerFieldKnnVectorsFormat(
        final Optional<MapperService> mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(
            mapperService,
            Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            Lucene99HnswVectorsFormat::new,
            new LuceneCodecFormatResolver(buildLuceneFormatResolvers()),
            new FaissCodecFormatResolver(mapperService, nativeIndexBuildStrategyFactory),
            nativeIndexBuildStrategyFactory
        );
    }

    private static Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> buildLuceneFormatResolvers() {
        return Map.of(LuceneVectorsFormatType.HNSW, ctx -> {
            final KNNVectorsFormatParams p = new KNNVectorsFormatParams(
                ctx.getParams(),
                ctx.getDefaultMaxConnections(),
                ctx.getDefaultBeamWidth(),
                ctx.getMethodContext().getSpaceType()
            );
            final Tuple<Integer, ExecutorService> merge = getMergeThreadCountAndExecutorService();
            if (p.getSpaceType() == SpaceType.HAMMING) {
                return new KNN9120HnswBinaryVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2());
            }
            return new Lucene99HnswVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2());
        }, LuceneVectorsFormatType.SCALAR_QUANTIZED, ctx -> {
            final KNNScalarQuantizedVectorsFormatParams p = new KNNScalarQuantizedVectorsFormatParams(
                ctx.getParams(),
                ctx.getDefaultMaxConnections(),
                ctx.getDefaultBeamWidth()
            );
            final Tuple<Integer, ExecutorService> merge = getMergeThreadCountAndExecutorService();
            return new Lucene99RWHnswScalarQuantizedVectorsFormat(
                p.getMaxConnections(),
                p.getBeamWidth(),
                merge.v1(),
                p.getBits(),
                p.isCompressFlag(),
                p.getConfidenceInterval(),
                merge.v2()
            );
        },
            LuceneVectorsFormatType.FLAT,
            ctx -> new Lucene104ScalarQuantizedVectorsFormat(Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE)
        );
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    private static Tuple<Integer, ExecutorService> getMergeThreadCountAndExecutorService() {
        int threadCount = mergeThreadCount;
        if (threadCount <= 1) {
            return DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE;
        }
        return Tuple.tuple(threadCount, mergeExecutor);
    }
}
