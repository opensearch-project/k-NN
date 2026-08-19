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
     * the encoder is sq with bits=1, otherwise falls back to the default native format.
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
        if (resolvedSpec.isFaissSQOneBit()) {
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
