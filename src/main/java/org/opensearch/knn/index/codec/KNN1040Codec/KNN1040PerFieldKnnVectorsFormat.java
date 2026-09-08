/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99RWHnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;

import org.opensearch.common.collect.Tuple;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
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
import org.opensearch.knn.index.engine.lucene.LuceneSQEncoder;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

/**
 * Per-field KNN vectors format for the KNN1040 codec. Uses {@link Lucene99HnswVectorsFormat}
 * for HNSW, {@link Lucene99RWHnswScalarQuantizedVectorsFormat} for scalar quantization (to
 * preserve the {@code confidenceInterval} parameter), and
 * {@link Lucene104ScalarQuantizedVectorsFormat} with {@code SINGLE_BIT_QUERY_NIBBLE} encoding
 * for the flat SQ method.
 */
public class KNN1040PerFieldKnnVectorsFormat extends KNN1040BasePerFieldKnnVectorsFormat {

    private static final Tuple<Integer, ExecutorService> DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE = Tuple.tuple(1, null);

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
            new LuceneCodecFormatResolver(buildLuceneFormatResolvers(), mapperService.orElse(null)),
            new FaissCodecFormatResolver(mapperService.orElse(null), nativeIndexBuildStrategyFactory),
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Maps the {@code index.knn.advanced.approximate_threshold} setting value to the
     * {@code tinySegmentsThreshold} used by Lucene HNSW writers.
     * <ul>
     *   <li>{@code approximateThreshold < 0} (e.g. {@code -1}) → {@link Integer#MAX_VALUE} (never build the graph)</li>
     *   <li>{@code approximateThreshold >= 0} → returned as-is (0 = always build, N = skip when docCount &lt; N)</li>
     * </ul>
     */
    static int toTinySegmentsThreshold(int approximateThreshold) {
        if (approximateThreshold < 0) {
            return Integer.MAX_VALUE;
        }
        return approximateThreshold;
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
            final int threshold = toTinySegmentsThreshold(ctx.getApproximateThreshold());
            if (p.getSpaceType() == SpaceType.HAMMING) {
                return new KNN9120HnswBinaryVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2(), threshold);
            }
            return new Lucene99HnswVectorsFormat(p.getMaxConnections(), p.getBeamWidth(), merge.v1(), merge.v2(), threshold);
        }, LuceneVectorsFormatType.SCALAR_QUANTIZED, ctx -> {
            final KNNScalarQuantizedVectorsFormatParams p = new KNNScalarQuantizedVectorsFormatParams(
                ctx.getParams(),
                ctx.getDefaultMaxConnections(),
                ctx.getDefaultBeamWidth()
            );
            final Tuple<Integer, ExecutorService> merge = getMergeThreadCountAndExecutorService();
            final int threshold = toTinySegmentsThreshold(ctx.getApproximateThreshold());
            if (p.getBits() == LuceneSQEncoder.Bits.ONE.getValue()) {
                return new KNN1040HnswScalarQuantizedVectorsFormat(
                    p.getBitEncoding(),
                    p.getMaxConnections(),
                    p.getBeamWidth(),
                    merge.v1(),
                    merge.v2(),
                    threshold
                );
            }
            return new Lucene99RWHnswScalarQuantizedVectorsFormat(
                p.getMaxConnections(),
                p.getBeamWidth(),
                merge.v1(),
                p.getBits(),
                p.isCompressFlag(),
                p.getConfidenceInterval(),
                merge.v2(),
                threshold
            );
        },
            LuceneVectorsFormatType.FLAT,
            ctx -> new KNN1040ScalarQuantizedVectorsFormat(resolveFlatScalarEncoding(ctx.getCompressionLevel()))
        );
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    private static Tuple<Integer, ExecutorService> getMergeThreadCountAndExecutorService() {
        int mergeThreadCount = KNNSettings.getIndexThreadQty();
        if (mergeThreadCount <= 1) {
            return DEFAULT_MERGE_THREAD_COUNT_AND_EXECUTOR_SERVICE;
        }
        return Tuple.tuple(mergeThreadCount, Executors.newFixedThreadPool(mergeThreadCount));
    }

    /**
     * Picks the {@link ScalarEncoding} for the FLAT format from the field's compression level.
     * x32 → 1-bit ({@code SINGLE_BIT_QUERY_NIBBLE}), x16 → 2-bit ({@code DIBIT_QUERY_NIBBLE}),
     * x8 → 4-bit ({@code PACKED_NIBBLE}). Any other value (including {@code NOT_CONFIGURED},
     * which the resolver maps to x32) falls back to 1-bit. {@link LuceneFlatMethodResolver}
     * rejects unsupported compression levels at mapping time so an unexpected value here would
     * indicate an upstream invariant violation.
     */
    private static ScalarEncoding resolveFlatScalarEncoding(final CompressionLevel compressionLevel) {
        if (compressionLevel == CompressionLevel.x8) {
            return ScalarEncodingResolver.forDocBits(4);
        }
        if (compressionLevel == CompressionLevel.x16) {
            return ScalarEncodingResolver.forDocBits(2);
        }
        return ScalarEncodingResolver.forDocBits(1);
    }
}
