/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN990Codec;

import org.apache.lucene.backward_codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN990PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
    private static final int NUM_MERGE_WORKERS = 1;

    public KNN990PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        super(
                mapperService,
                Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
                Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
                Lucene99HnswVectorsFormat::new,
                buildLuceneFormatResolvers(),
                new NativeIndexBuildStrategyFactory());
    }

    private static Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> buildLuceneFormatResolvers() {
        return Map.of(
                LuceneVectorsFormatType.HNSW, ctx -> {
                    final KNNVectorsFormatParams p = new KNNVectorsFormatParams(
                            ctx.getParams(), ctx.getDefaultMaxConnections(), ctx.getDefaultBeamWidth(),
                            ctx.getMethodContext().getSpaceType());
                    return new Lucene99HnswVectorsFormat(p.getMaxConnections(), p.getBeamWidth());
                },
                LuceneVectorsFormatType.SCALAR_QUANTIZED, ctx -> {
                    final KNNScalarQuantizedVectorsFormatParams p = new KNNScalarQuantizedVectorsFormatParams(
                            ctx.getParams(), ctx.getDefaultMaxConnections(), ctx.getDefaultBeamWidth());
                    return new Lucene99HnswScalarQuantizedVectorsFormat(
                            p.getMaxConnections(), p.getBeamWidth(), NUM_MERGE_WORKERS, p.getBits(), p.isCompressFlag(),
                            p.getConfidenceInterval(), null);
                });
    }

    @Override
    /**
     * This method returns the maximum dimension allowed from KNNEngine for Lucene
     * codec
     *
     * @param fieldName Name of the field, ignored
     * @return Maximum constant dimension set by KNNEngine
     */
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }
}
