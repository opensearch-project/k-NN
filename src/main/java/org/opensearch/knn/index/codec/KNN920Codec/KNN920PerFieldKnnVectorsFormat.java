/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN920Codec;

import org.apache.lucene.backward_codecs.lucene92.Lucene92HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;

import java.util.Optional;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN920PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {

    public KNN920PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        super(
            mapperService,
            Lucene92HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene92HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            () -> new Lucene92HnswVectorsFormat(),
            knnVectorsFormatParams -> new Lucene92HnswVectorsFormat(
                knnVectorsFormatParams.getMaxConnections(),
                knnVectorsFormatParams.getBeamWidth()
            )
        );
    }
}
