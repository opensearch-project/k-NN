/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

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
            buildLuceneFormatResolvers(),
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
            // There is an assumption here that hamming space will only be used for binary
            // vectors.
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

    /**
     * This method returns the maximum dimension allowed from KNNEngine for Lucene
     * codec
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
