/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;
import java.util.Optional;

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

    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    ) {
        throw new UnsupportedOperationException(
            String.format("%s does not require field context, use resolve() instead", getClass().getSimpleName())
        );
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
        // mapperService is already checked for null or valid instance type at caller,
        // hence we don't need additional isPresent check here.
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null
            ? approximateThresholdValue
            : KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }
}
