/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.BEAM_WIDTH;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MAX_CONNECTIONS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

/**
 * {@link CodecFormatResolver} implementation for the Lucene engine. Combines format type
 * determination logic with the format factory map to resolve the appropriate
 * {@link KnnVectorsFormat} for a given Lucene field.
 *
 * <p>The constructor accepts a format factory map so that codec subclasses can provide
 * codec-specific Lucene format factories.</p>
 */
@Log4j2
public class LuceneCodecFormatResolver implements CodecFormatResolver {

    private final Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> formatResolvers;
    private final Optional<MapperService> mapperService;

    public LuceneCodecFormatResolver(
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> formatResolvers,
        Optional<MapperService> mapperService
    ) {
        this.formatResolvers = formatResolvers;
        this.mapperService = mapperService;
    }

    /**
     * Backward-compatible constructor that does not wire in the {@code approximate_threshold} setting.
     * The resolver will fall back to the default threshold value.
     */
    public LuceneCodecFormatResolver(Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> formatResolvers) {
        this(formatResolvers, Optional.empty());
    }

    @Override
    public KnnVectorsFormat resolve() {
        throw new UnsupportedOperationException(
            String.format("%s requires field context, use resolve(field, ...) instead", getClass().getSimpleName())
        );
    }

    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    ) {
        LuceneVectorsFormatType formatType = determineFormatType(field, methodContext, params, defaultMaxConnections, defaultBeamWidth);
        Function<KnnVectorsFormatContext, KnnVectorsFormat> factory = formatResolvers.get(formatType);
        if (factory == null) {
            throw new IllegalStateException(String.format("No Lucene vectors format registered for type [%s]", formatType));
        }
        final int approximateThreshold = getApproximateThresholdValue();
        return factory.apply(
            new KnnVectorsFormatContext(field, methodContext, params, defaultMaxConnections, defaultBeamWidth, approximateThreshold)
        );
    }

    /**
     * Retrieves the approximate threshold value from index settings.
     * Falls back to the default value when the mapper service is unavailable or the setting is not
     * explicitly configured.
     */
    private int getApproximateThresholdValue() {
        if (mapperService.isEmpty()) {
            return KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
        }
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null
            ? approximateThresholdValue
            : KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }

    /**
     * Determines the {@link LuceneVectorsFormatType} based on the method context and parameters.
     * Moved from {@code KNN1040BasePerFieldKnnVectorsFormat.resolveLuceneFormat}.
     *
     * <ul>
     *   <li>Flat method name → {@link LuceneVectorsFormatType#FLAT}</li>
     *   <li>Encoder parameter with valid SQ config → {@link LuceneVectorsFormatType#SCALAR_QUANTIZED}</li>
     *   <li>HNSW without encoder → {@link LuceneVectorsFormatType#HNSW}</li>
     * </ul>
     */
    private LuceneVectorsFormatType determineFormatType(
        final String field,
        final KNNMethodContext methodContext,
        final Map<String, Object> params,
        final int defaultMaxConnections,
        final int defaultBeamWidth
    ) {
        if (METHOD_FLAT.equals(methodContext.getMethodComponentContext().getName())) {
            log.debug("Initialize KNN vector format for field [{}] with Lucene SQ flat format", field);
            return LuceneVectorsFormatType.FLAT;
        }

        if (params != null && params.containsKey(METHOD_ENCODER_PARAMETER)) {
            KNNScalarQuantizedVectorsFormatParams sqParams = new KNNScalarQuantizedVectorsFormatParams(
                params,
                defaultMaxConnections,
                defaultBeamWidth
            );
            if (sqParams.validate(params)) {
                log.debug(
                    "Initialize KNN vector format for field [{}] with params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\"",
                    field,
                    MAX_CONNECTIONS,
                    sqParams.getMaxConnections(),
                    BEAM_WIDTH,
                    sqParams.getBeamWidth(),
                    LUCENE_SQ_CONFIDENCE_INTERVAL,
                    sqParams.getConfidenceInterval(),
                    LUCENE_SQ_BITS,
                    sqParams.getBits()
                );
                return LuceneVectorsFormatType.SCALAR_QUANTIZED;
            }
        }

        log.debug(
            "Initialize KNN vector format for field [{}] with params [{}] = \"{}\" and [{}] = \"{}\"",
            field,
            MAX_CONNECTIONS,
            defaultMaxConnections,
            BEAM_WIDTH,
            defaultBeamWidth
        );
        return LuceneVectorsFormatType.HNSW;
    }
}
