/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KnnVectorsFormatContext;
import org.opensearch.knn.index.codec.LuceneVectorsFormatType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * Unit tests for {@link LuceneCodecFormatResolver}.
 * Validates format type determination (FLAT, SCALAR_QUANTIZED, HNSW),
 * error handling for missing factories, and correct context construction.
 */
public class LuceneCodecFormatResolverTests extends KNNTestCase {

    private static final String TEST_FIELD = "test_vector";
    private static final int DEFAULT_MAX_CONN = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    private static final KnnVectorsFormat HNSW_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat SQ_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat FLAT_FORMAT = mock(KnnVectorsFormat.class);

    public void testResolve_whenFlatMethod_thenReturnFlatFormat() {
        KNNMethodContext flatContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.FLAT,
            ctx -> FLAT_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, flatContext, Collections.emptyMap(), DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertSame(FLAT_FORMAT, result);
    }

    public void testResolve_whenHnswWithSQEncoder_thenReturnSQFormat() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNMethodContext sqContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.SCALAR_QUANTIZED,
            ctx -> SQ_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, sqContext, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertSame(SQ_FORMAT, result);
    }

    public void testResolve_whenHnswWithoutEncoder_thenReturnHnswFormat() {
        KNNMethodContext hnswContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 32, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        Map<String, Object> params = Map.of(METHOD_PARAMETER_M, 32, METHOD_PARAMETER_EF_CONSTRUCTION, 256);
        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, hnswContext, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertSame(HNSW_FORMAT, result);
    }

    public void testResolve_whenFormatTypeNotRegistered_thenThrowIllegalStateException() {
        KNNMethodContext flatContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );

        // Only register HNSW, but flat method will resolve to FLAT type
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        IllegalStateException ex = expectThrows(
            IllegalStateException.class,
            () -> resolver.resolve(TEST_FIELD, flatContext, Collections.emptyMap(), DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH)
        );
        assertTrue(ex.getMessage().contains("FLAT"));
    }

    public void testResolve_contextContainsCorrectValues() {
        int customM = 64;
        int customEf = 512;
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_PARAMETER_M, customM);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, customEf);

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            new MethodComponentContext(METHOD_HNSW, params)
        );

        final KnnVectorsFormatContext[] capturedContext = new KnnVectorsFormatContext[1];
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> {
                capturedContext[0] = ctx;
                return HNSW_FORMAT;
            }
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        resolver.resolve(TEST_FIELD, methodContext, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);

        assertNotNull(capturedContext[0]);
        assertEquals(TEST_FIELD, capturedContext[0].getField());
        assertSame(methodContext, capturedContext[0].getMethodContext());
        assertSame(params, capturedContext[0].getParams());
        assertEquals(DEFAULT_MAX_CONN, capturedContext[0].getDefaultMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, capturedContext[0].getDefaultBeamWidth());
    }

    public void testResolve_whenHnswWithNullParams_thenReturnHnswFormat() {
        KNNMethodContext hnswContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, hnswContext, null, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertSame(HNSW_FORMAT, result);
    }

    public void testResolve_whenNoArgCalled_thenThrowUnsupportedOperationException() {
        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(Map.of());
        expectThrows(UnsupportedOperationException.class, resolver::resolve);
    }

    public void testResolve_whenHnswWithSQOneBitEncoder_thenReturnSQFormat() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(LUCENE_SQ_BITS, 1);
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNMethodContext sqContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.SCALAR_QUANTIZED,
            ctx -> SQ_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, sqContext, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertSame("SQ with bits=1 should route to SCALAR_QUANTIZED resolver", SQ_FORMAT, result);
    }

    public void testResolve_whenHnswWithSQOneBitEncoder_thenContextContainsBitsParam() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(LUCENE_SQ_BITS, 1);
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 32);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 200);

        KNNMethodContext sqContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));

        final KnnVectorsFormatContext[] capturedContext = new KnnVectorsFormatContext[1];
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.SCALAR_QUANTIZED,
            ctx -> {
                capturedContext[0] = ctx;
                return SQ_FORMAT;
            },
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        LuceneCodecFormatResolver resolver = new LuceneCodecFormatResolver(resolvers);
        resolver.resolve(TEST_FIELD, sqContext, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);

        assertNotNull(capturedContext[0]);
        assertEquals(TEST_FIELD, capturedContext[0].getField());
        Object encoder = capturedContext[0].getParams().get(METHOD_ENCODER_PARAMETER);
        assertTrue(encoder instanceof MethodComponentContext);
        assertEquals(1, ((MethodComponentContext) encoder).getParameters().get(LUCENE_SQ_BITS));
    }
}
