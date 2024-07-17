/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.BEAM_WIDTH;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_COMPRESS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.MAX_CONNECTIONS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Base class for PerFieldKnnVectorsFormat, builds KnnVectorsFormat based on specific Lucene version
 */
@AllArgsConstructor
@Log4j2
public abstract class BasePerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> mapperService;
    private final int defaultMaxConnections;
    private final int defaultBeamWidth;
    private final Supplier<KnnVectorsFormat> defaultFormatSupplier;
    private final BiFunction<Integer, Integer, KnnVectorsFormat> formatSupplier;
    private final Function5Arity<Integer, Integer, Float, Integer, Boolean, KnnVectorsFormat> quantizedVectorsFormatSupplier;

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(final String field) {
        if (isKnnVectorFieldType(field) == false) {
            log.debug(
                "Initialize KNN vector format for field [{}] with default params [max_connections] = \"{}\" and [beam_width] = \"{}\"",
                field,
                defaultMaxConnections,
                defaultBeamWidth
            );
            return defaultFormatSupplier.get();
        }
        var type = (KNNVectorFieldMapper.KNNVectorFieldType) mapperService.orElseThrow(
            () -> new IllegalStateException(
                String.format("Cannot read field type for field [%s] because mapper service is not available", field)
            )
        ).fieldType(field);
        var params = type.getKnnMethodContext().getMethodComponentContext().getParameters();
        int maxConnections = getMaxConnections(params);
        int beamWidth = getBeamWidth(params);

        if (type.getKnnMethodContext().getKnnEngine() == KNNEngine.LUCENE
            && params != null
            && params.containsKey(METHOD_ENCODER_PARAMETER)) {
            final KnnVectorsFormat knnVectorsFormat = validateAndApplyQuantizedVectorsFormatForLuceneEngine(
                params,
                field,
                maxConnections,
                beamWidth
            );
            if (knnVectorsFormat != null) {
                return knnVectorsFormat;
            }
        }

        log.debug(
            "Initialize KNN vector format for field [{}] with params [max_connections] = \"{}\" and [beam_width] = \"{}\"",
            field,
            maxConnections,
            beamWidth
        );
        return formatSupplier.apply(maxConnections, beamWidth);
    }

    private KnnVectorsFormat validateAndApplyQuantizedVectorsFormatForLuceneEngine(
        final Map<String, Object> params,
        final String field,
        final int maxConnections,
        final int beamWidth
    ) {

        if (params.get(METHOD_ENCODER_PARAMETER) == null) {
            return null;
        }

        // Validate if the object is of type MethodComponentContext before casting it later
        if (!(params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            return null;
        }
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        if (!ENCODER_SQ.equals(encoderMethodComponentContext.getName())) {
            return null;
        }
        Map<String, Object> sqEncoderParams = encoderMethodComponentContext.getParameters();
        Float confidenceInterval = getConfidenceInterval(sqEncoderParams);
        int bits = getBits(sqEncoderParams);
        boolean compressFlag = getCompressFlag(sqEncoderParams);
        log.debug(
            "Initialize KNN vector format for field [{}] with params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\",[{}] = \"{}\"",
            field,
            MAX_CONNECTIONS,
            maxConnections,
            BEAM_WIDTH,
            beamWidth,
            LUCENE_SQ_CONFIDENCE_INTERVAL,
            confidenceInterval,
            LUCENE_SQ_BITS,
            bits,
            LUCENE_SQ_COMPRESS,
            compressFlag
        );
        return quantizedVectorsFormatSupplier.apply(maxConnections, beamWidth, confidenceInterval, bits, compressFlag);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return mapperService.isPresent() && mapperService.get().fieldType(field) instanceof KNNVectorFieldMapper.KNNVectorFieldType;
    }

    private int getMaxConnections(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_M);
        }
        return defaultMaxConnections;
    }

    private int getBeamWidth(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
        }
        return defaultBeamWidth;
    }

    private Float getConfidenceInterval(final Map<String, Object> params) {

        if (params != null && params.containsKey(LUCENE_SQ_CONFIDENCE_INTERVAL)) {
            if (params.get("confidence_interval").equals(0)) return Float.valueOf(0);

            return ((Double) params.get("confidence_interval")).floatValue();

        }
        return null;
    }

    private int getBits(final Map<String, Object> params) {
        if (params != null && params.containsKey(LUCENE_SQ_BITS)) {
            return (int) params.get("bits");
        }
        return LUCENE_SQ_DEFAULT_BITS;
    }

    private boolean getCompressFlag(final Map<String, Object> params) {
        if (params != null && params.containsKey(LUCENE_SQ_COMPRESS)) {
            return (boolean) params.get("compress");
        }
        return false;
    }
}
