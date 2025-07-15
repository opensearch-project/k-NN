/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

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
            },// TODO: this is the format supplier. may be able to get from mapper service, if it's bbq or not.
            knnBBQVectorsFormatParams -> {
                final Tuple<Integer, ExecutorService> mergeThreadCountAndExecutorService = getMergeThreadCountAndExecutorService();
                return new Lucene102HnswBinaryQuantizedVectorsFormat(
                        knnBBQVectorsFormatParams.getMaxConnections(),
                        knnBBQVectorsFormatParams.getBeamWidth(),
                        mergeThreadCountAndExecutorService.v1(),
                        mergeThreadCountAndExecutorService.v2()
                );
            },
            knnScalarQuantizedVectorsFormatParams -> {
                final Tuple<Integer, ExecutorService> mergeThreadCountAndExecutorService = getMergeThreadCountAndExecutorService();
                return new Lucene99HnswScalarQuantizedVectorsFormat(
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
        // To ensure that only once we are fetching the settings per segment, we are fetching the num threads once while
        // creating the executors
        int mergeThreadCount = KNNSettings.getIndexThreadQty();
        // We need to return null whenever the merge threads are <=1, as lucene assumes that if number of threads are 1
        // then we should be giving a null value of the executor
        if (mergeThreadCount <= 1) {
            return DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE;
        } else {
            return Tuple.tuple(mergeThreadCount, Executors.newFixedThreadPool(mergeThreadCount));
        }
    }
}
