/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.common.Nullable;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;

import java.util.Map;

/**
 * {@link CodecFormatResolver} implementation for native engines (FAISS, NMSLIB).
 * Encapsulates the {@link NativeEngines990KnnVectorsFormat} creation logic including
 * {@code approximateThreshold} lookup from index settings.
 *
 * <p>Placed in the {@code faiss} package alongside {@link FaissMethodResolver} because
 * NMSLIB is deprecated and no new NMSLIB indices are created.</p>
 */
public class FaissCodecFormatResolver implements CodecFormatResolver {

    @Nullable
    private final MapperService mapperService;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public FaissCodecFormatResolver(
        @Nullable MapperService mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.mapperService = mapperService;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Resolves the format for a specific field. Returns {@link Faiss1040ScalarQuantizedKnnVectorsFormat} when
     * the encoder is sq with bits in {1, 2, 4} (the multi-bit MOS path), otherwise falls back to the default
     * native format. The document bit width is resolved per-field at write time by the format itself from
     * {@code SQ_CONFIG}, so the format instance stays encoding-agnostic and SPI-instantiated instances
     * cannot silently miswrite fields.
     */
    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        ResolvedIndexSpec resolvedSpec
    ) {
        if (resolvedSpec.isFaissSQMultiBit()) {
            return new Faiss1040ScalarQuantizedKnnVectorsFormat(
                KNNSettings.getApproximateThresholdValue(mapperService),
                nativeIndexBuildStrategyFactory
            );
        }
        return resolve();
    }

    @Override
    public KnnVectorsFormat resolve() {
        final int approximateThreshold = KNNSettings.getApproximateThresholdValue(mapperService);
        return new NativeEngines990KnnVectorsFormat(approximateThreshold, nativeIndexBuildStrategyFactory);
    }
}
