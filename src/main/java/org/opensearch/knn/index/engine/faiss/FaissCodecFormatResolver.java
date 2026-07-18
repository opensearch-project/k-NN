/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.index.engine.faiss.FaissSQEncoder.isSQMultiBit;

/**
 * {@link CodecFormatResolver} implementation for native engines (FAISS, NMSLIB).
 * Encapsulates the {@link NativeEngines990KnnVectorsFormat} creation logic including
 * {@code approximateThreshold} lookup from index settings.
 *
 * <p>Placed in the {@code faiss} package alongside {@link FaissMethodResolver} because
 * NMSLIB is deprecated and no new NMSLIB indices are created.</p>
 */
public class FaissCodecFormatResolver implements CodecFormatResolver {

    private final Optional<MapperService> mapperService;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public FaissCodecFormatResolver(
        Optional<MapperService> mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.mapperService = mapperService;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Resolves the format for a specific field. Returns {@link Faiss1040ScalarQuantizedKnnVectorsFormat} when
     * the encoder is sq with bits in {1, 2, 4} (the multi-bit MOS path), otherwise falls back to the default
     * native format. The document bit width is resolved per-field at write time by the format itself.
     */
    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    ) {
        if (isSQMultiBit(params)) {
            // Encoding is not passed here — Faiss1040ScalarQuantizedKnnVectorsFormat resolves it
            // per-field from SQ_CONFIG at fieldsWriter() time. This keeps the format instance
            // encoding-agnostic so SPI-instantiated instances cannot silently miswrite fields.
            return new Faiss1040ScalarQuantizedKnnVectorsFormat(nativeIndexBuildStrategyFactory);
        }
        return resolve();
    }

    @Override
    public KnnVectorsFormat resolve() {
        final int approximateThreshold = getApproximateThresholdValue();
        return new NativeEngines990KnnVectorsFormat(approximateThreshold, nativeIndexBuildStrategyFactory);
    }

    /**
     * Retrieves the approximate threshold value from index settings.
     * Falls back to the default value when the setting is not explicitly configured.
     */
    private int getApproximateThresholdValue() {
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null
            ? approximateThresholdValue
            : KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }

}
