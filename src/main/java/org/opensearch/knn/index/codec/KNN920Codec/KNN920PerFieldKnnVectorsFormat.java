/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.backward_codecs.lucene92.Lucene92HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.Map;
import java.util.Optional;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
@AllArgsConstructor
@Log4j2
public class KNN920PerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

    private final Optional<MapperService> mapperService;

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(final String field) {
        if (isNotKnnVectorFieldType(field)) {
            log.debug(
                String.format(
                    "Initialize KNN vector format for field [%s] with default params [max_connections] = \"%d\" and [beam_width] = \"%d\"",
                    field,
                    Lucene92HnswVectorsFormat.DEFAULT_MAX_CONN,
                    Lucene92HnswVectorsFormat.DEFAULT_BEAM_WIDTH
                )
            );
            return new Lucene92HnswVectorsFormat();
        }
        var type = (KNNVectorFieldMapper.KNNVectorFieldType) mapperService.orElseThrow(
            () -> new IllegalStateException(
                String.format("Cannot read field type for field [%s] because mapper service is not available", field)
            )
        ).fieldType(field);
        var params = type.getKnnMethodContext().getMethodComponent().getParameters();
        int maxConnections = getMaxConnections(params);
        int beamWidth = getBeamWidth(params);
        log.debug(
            String.format(
                "Initialize KNN vector format for field [%s] with params [max_connections] = \"%d\" and [beam_width] = \"%d\"",
                field,
                maxConnections,
                beamWidth
            )
        );
        return new Lucene92HnswVectorsFormat(maxConnections, beamWidth);
    }

    private boolean isNotKnnVectorFieldType(final String field) {
        return !mapperService.isPresent() || !(mapperService.get().fieldType(field) instanceof KNNVectorFieldMapper.KNNVectorFieldType);
    }

    private int getMaxConnections(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_M);
        }
        return Lucene92HnswVectorsFormat.DEFAULT_MAX_CONN;
    }

    private int getBeamWidth(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
        }
        return Lucene92HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
    }
}
