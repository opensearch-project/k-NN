/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN940Codec;

import org.apache.lucene.backward_codecs.lucene94.Lucene94HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN940PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {

    public KNN940PerFieldKnnVectorsFormat(final MapperService mapperService) {
        super(
            mapperService,
            Lucene94HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene94HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            Lucene94HnswVectorsFormat::new,
            knnVectorsFormatParams -> new Lucene94HnswVectorsFormat(
                knnVectorsFormatParams.getMaxConnections(),
                knnVectorsFormatParams.getBeamWidth()
            )
        );
    }
}
