/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN920Codec;

import org.apache.lucene.backward_codecs.lucene92.Lucene92HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;

import java.util.Map;
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
                Map.of(
                        LuceneVectorsFormatType.HNSW, ctx -> {
                            final KNNVectorsFormatParams p = new KNNVectorsFormatParams(
                                    ctx.getParams(), ctx.getDefaultMaxConnections(), ctx.getDefaultBeamWidth(),
                                    ctx.getMethodContext().getSpaceType());
                            return new Lucene92HnswVectorsFormat(p.getMaxConnections(), p.getBeamWidth());
                        }),
                new NativeIndexBuildStrategyFactory());
    }
}
