/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.backward_codecs.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120HnswBinaryVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN9120PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
    private static final Tuple<Integer, ExecutorService> DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE = Tuple.tuple(1, null);

    public KNN9120PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        this(mapperService, new NativeIndexBuildStrategyFactory());
    }

    public KNN9120PerFieldKnnVectorsFormat(
        final Optional<MapperService> mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(
            mapperService,
            Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            Lucene99HnswVectorsFormat::new,
            knnVectorsFormatParams -> {
                final Tuple<Integer, ExecutorService> mergeThreadCountAndExecutorService = getMergeThreadCountAndExecutorService();
                // There is an assumption here that hamming space will only be used for binary vectors. This will need to be fixed if that
                // changes in the future.
                if (knnVectorsFormatParams.getSpaceType() == SpaceType.HAMMING) {
                    return new KNN9120HnswBinaryVectorsFormat(
                        knnVectorsFormatParams.getMaxConnections(),
                        knnVectorsFormatParams.getBeamWidth(),
                        // number of merge threads
                        mergeThreadCountAndExecutorService.v1(),
                        // executor service
                        mergeThreadCountAndExecutorService.v2()
                    );
                } else {
                    return new Lucene99HnswVectorsFormat(
                        knnVectorsFormatParams.getMaxConnections(),
                        knnVectorsFormatParams.getBeamWidth(),
                        // number of merge threads
                        mergeThreadCountAndExecutorService.v1(),
                        // executor service
                        mergeThreadCountAndExecutorService.v2()
                    );
                }
            },
            knnScalarQuantizedVectorsFormatParams -> {
                final Tuple<Integer, ExecutorService> mergeThreadCountAndExecutorService = getMergeThreadCountAndExecutorService();
                return new Lucene99RWHnswScalarQuantizedVectorsFormat(
                    knnScalarQuantizedVectorsFormatParams.getMaxConnections(),
                    knnScalarQuantizedVectorsFormatParams.getBeamWidth(),
                    // Number of merge threads
                    mergeThreadCountAndExecutorService.v1(),
                    knnScalarQuantizedVectorsFormatParams.getBits(),
                    knnScalarQuantizedVectorsFormatParams.isCompressFlag(),
                    knnScalarQuantizedVectorsFormatParams.getConfidenceInterval(),
                    // Executor service
                    mergeThreadCountAndExecutorService.v2()
                );
            },
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * This method returns the maximum dimension allowed from KNNEngine for Lucene codec
     *
     * @param fieldName Name of the field, ignored
     * @return Maximum constant dimension set by KNNEngine
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    private static Tuple<Integer, ExecutorService> getMergeThreadCountAndExecutorService() {
        int mergeThreadCount = KNNSettings.getIndexThreadQty();
        return buildMergeThreadCountAndExecutorService(mergeThreadCount);
    }

    static Tuple<Integer, ExecutorService> buildMergeThreadCountAndExecutorService(int mergeThreadCount) {
        // Lucene expects a null executor when the merge thread count is <= 1
        if (mergeThreadCount <= 1) {
            return DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE;
        }
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(mergeThreadCount);
        executor.setKeepAliveTime(60L, TimeUnit.SECONDS);
        executor.allowCoreThreadTimeOut(true);
        return Tuple.tuple(mergeThreadCount, executor);
    }
}
