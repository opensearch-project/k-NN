/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.KNN920Codec;

import lombok.Builder;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene92.Lucene92HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.KNNFormatFactory;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN92DefaultDelegate;

/**
 * KNN codec that is based on Lucene92 codec
 */
@Log4j2
public final class KNN920Codec extends FilterCodec {

    private static final String KNN920 = "KNN920Codec";

    private final KNNFormatFacade knnFormatFacade;
    private final Optional<MapperService> mapperService;
    private final KNN920PerFieldKnnVectorsFormat perFieldKnnVectorsFormat;

    /**
     * No arg constructor that uses Lucene91 as the delegate
     */
    public KNN920Codec() {
        this(createKNN92DefaultDelegate(), Optional.empty());
    }

    /**
     * Constructor that takes a Codec delegate to delegate all methods this code does not implement to.
     *
     * @param delegate codec that will perform all operations this codec does not override
     */
    @Builder
    public KNN920Codec(Codec delegate, Optional<MapperService> mapperService) {
        super(KNN920, delegate);
        this.mapperService = mapperService;
        knnFormatFacade = KNNFormatFactory.createKNN920Format(delegate);
        perFieldKnnVectorsFormat = new KNN920PerFieldKnnVectorsFormat();
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return knnFormatFacade.docValuesFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return knnFormatFacade.compoundFormat();
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return perFieldKnnVectorsFormat;
    }

    /**
     * Class provides per field format implementation for Lucene Knn vector type
     */
    class KNN920PerFieldKnnVectorsFormat extends PerFieldKnnVectorsFormat {

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
            final KNNVectorFieldMapper.KNNVectorFieldType type = (KNNVectorFieldMapper.KNNVectorFieldType) mapperService.orElseThrow(
                () -> new IllegalStateException(
                    String.format("Cannot read field type for field [%s] because mapper service is not available", field)
                )
            ).fieldType(field);
            final Map<String, Object> params = type.getKnnMethodContext().getMethodComponent().getParameters();
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
            var luceneHnswVectorsFormat = new Lucene92HnswVectorsFormat(maxConnections, beamWidth);
            return luceneHnswVectorsFormat;
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
}
