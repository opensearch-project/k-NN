/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN950Codec;

import org.apache.lucene.backward_codecs.lucene95.Lucene95HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Optional;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN950PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {

    public KNN950PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        super(
            mapperService,
            Lucene95HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene95HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            () -> new Lucene95HnswVectorsFormat(),
            knnVectorsFormatParams -> new Lucene95HnswVectorsFormat(
                knnVectorsFormatParams.getMaxConnections(),
                knnVectorsFormatParams.getBeamWidth()
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
