/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.codec.params.KNN1040ScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Map;
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

    public LuceneCodecFormatResolver(Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> formatResolvers) {
        this.formatResolvers = formatResolvers;
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
        return factory.apply(new KnnVectorsFormatContext(field, methodContext, params, defaultMaxConnections, defaultBeamWidth));
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
            log.debug("Initialize KNN vector format for field [{}] with Lucene BBQ flat format", field);
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
            } else {
                // Temporary route for BBQ - eventually this should be used for SQ as well
                KNN1040ScalarQuantizedVectorsFormatParams sqBBQParams = new KNN1040ScalarQuantizedVectorsFormatParams(
                    params,
                    defaultMaxConnections,
                    defaultBeamWidth
                );
                if (sqBBQParams.validate(params)) {
                    log.debug(
                        "Initialize KNN vector format for field [{}] with scalar/binary quantization, params [{}] = \"{}\", [{}] = \"{}\"",
                        field,
                        MAX_CONNECTIONS,
                        sqBBQParams.getMaxConnections(),
                        BEAM_WIDTH,
                        sqBBQParams.getBeamWidth()
                    );
                    return LuceneVectorsFormatType.BBQ;
                }
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
