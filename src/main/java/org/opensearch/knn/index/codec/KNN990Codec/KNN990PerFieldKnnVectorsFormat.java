/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Optional;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN990PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {

    public KNN990PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        super(
            mapperService,
            Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            () -> new Lucene99HnswVectorsFormat(),
            (maxConnm, beamWidth) -> new Lucene99HnswVectorsFormat(maxConnm, beamWidth),
            (maxConnm, beamWidth, confidenceInterval, bits, compress) -> new Lucene99HnswScalarQuantizedVectorsFormat(
                maxConnm,
                beamWidth,
                1,
                bits,
                compress,
                confidenceInterval,
                null
            )
        );
    }

    @Override
    /**
     * This method returns the maximum dimension allowed from KNNEngine for Lucene codec
     *
     * @param fieldName Name of the field, ignored
     * @return Maximum constant dimension set by KNNEngine
     */
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }
}
