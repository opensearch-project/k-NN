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
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.BEAM_WIDTH;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_COMPRESS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
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
    private final Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier;
    private final Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> scalarQuantizedVectorsFormatSupplier;

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

        if (type.getKnnMethodContext().getKnnEngine() == KNNEngine.LUCENE
            && params != null
            && params.containsKey(METHOD_ENCODER_PARAMETER)) {
            KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams();
            if (knnScalarQuantizedVectorsFormatParams.validate(params)) {
                knnScalarQuantizedVectorsFormatParams.initialize(params, defaultMaxConnections, defaultBeamWidth);
                log.debug(
                    "Initialize KNN vector format for field [{}] with params [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\", [{}] = \"{}\",[{}] = \"{}\"",
                    field,
                    MAX_CONNECTIONS,
                    knnScalarQuantizedVectorsFormatParams.getMaxConnections(),
                    BEAM_WIDTH,
                    knnScalarQuantizedVectorsFormatParams.getBeamWidth(),
                    LUCENE_SQ_CONFIDENCE_INTERVAL,
                    knnScalarQuantizedVectorsFormatParams.getConfidenceInterval(),
                    LUCENE_SQ_BITS,
                    knnScalarQuantizedVectorsFormatParams.getBits(),
                    LUCENE_SQ_COMPRESS,
                    knnScalarQuantizedVectorsFormatParams.isCompressFlag()
                );
                return scalarQuantizedVectorsFormatSupplier.apply(knnScalarQuantizedVectorsFormatParams);
            }

        }

        KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams();
        knnVectorsFormatParams.initialize(params, defaultMaxConnections, defaultBeamWidth);
        log.debug(
            "Initialize KNN vector format for field [{}] with params [max_connections] = \"{}\" and [beam_width] = \"{}\"",
            field,
            knnVectorsFormatParams.getMaxConnections(),
            knnVectorsFormatParams.getBeamWidth()
        );
        return vectorsFormatSupplier.apply(knnVectorsFormatParams);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
    }

    private boolean isKnnVectorFieldType(final String field) {
        return mapperService.isPresent() && mapperService.get().fieldType(field) instanceof KNNVectorFieldMapper.KNNVectorFieldType;
    }
}
