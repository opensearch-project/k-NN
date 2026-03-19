/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.backward_codecs.BasePerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.faiss.FaissCodecFormatResolver;
import org.opensearch.knn.index.engine.lucene.LuceneCodecFormatResolver;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_OPTIMIZED_SCALAR_QUANTIZER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * Unit tests for {@link BasePerFieldKnnVectorsFormat} verifying the
 * registry-based
 * format resolution pattern.
 */
public class BasePerFieldKnnVectorsFormatTests extends KNNTestCase {

    private static final String TEST_FIELD = "test_vector";
    private static final int DEFAULT_MAX_CONN = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    // Sentinel format instances used to verify correct resolution
    private static final KnnVectorsFormat HNSW_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat SQ_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat OPTIMIZED_SCALAR_QUANTIZER_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat FLAT_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat DEFAULT_FORMAT = mock(KnnVectorsFormat.class);

    /**
     * Concrete subclass for testing the registry-based path via KNN1040BasePerFieldKnnVectorsFormat.
     */
    private static class TestPerFieldKnnVectorsFormat extends KNN1040BasePerFieldKnnVectorsFormat {
        TestPerFieldKnnVectorsFormat(
            Optional<MapperService> mapperService,
            Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers
        ) {
            super(
                mapperService,
                DEFAULT_MAX_CONN,
                DEFAULT_BEAM_WIDTH,
                () -> DEFAULT_FORMAT,
                new LuceneCodecFormatResolver(resolvers),
                new FaissCodecFormatResolver(mapperService, new NativeIndexBuildStrategyFactory()),
                new NativeIndexBuildStrategyFactory()
            );
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
        }
    }

    /**
     * Legacy concrete subclass for testing the legacy constructor path
     * (backward codecs KNN920–KNN9120) that uses vectorsFormatSupplier.
     */
    private static class LegacyTestPerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
        LegacyTestPerFieldKnnVectorsFormat(
            Optional<MapperService> mapperService,
            Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier
        ) {
            super(mapperService, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, () -> DEFAULT_FORMAT, vectorsFormatSupplier);
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
        }
    }

    /**
     * Legacy concrete subclass for testing the legacy constructor path with SQ support
     * (backward codecs like KNN990–KNN9120) that uses both vectorsFormatSupplier
     * and scalarQuantizedVectorsFormatSupplier.
     */
    private static class LegacySQTestPerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
        LegacySQTestPerFieldKnnVectorsFormat(
            Optional<MapperService> mapperService,
            Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier,
            Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> scalarQuantizedVectorsFormatSupplier
        ) {
            super(
                mapperService,
                DEFAULT_MAX_CONN,
                DEFAULT_BEAM_WIDTH,
                () -> DEFAULT_FORMAT,
                vectorsFormatSupplier,
                scalarQuantizedVectorsFormatSupplier
            );
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
        }
    }

    /**
     * When the field is not a KNN vector type, the default format should be
     * returned.
     */
    public void testGetKnnVectorsFormatForField_whenNotKnnField_thenReturnDefaultFormat() {
        // Empty mapperService means isKnnVectorFieldType returns false
        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.empty(), Map.of());

        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame(DEFAULT_FORMAT, result);
    }

    /**
     * When the Lucene engine is used with the HNSW method, the HNSW resolver should
     * be called.
     */
    public void testGetKnnVectorsFormatForField_whenLuceneHnsw_thenReturnHnswFormat() {
        KNNMethodContext hnswMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 32, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, hnswMethodContext);

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame(HNSW_FORMAT, result);
    }

    /**
     * When the Lucene engine is used with the flat method, the FLAT resolver should
     * be called.
     */
    public void testGetKnnVectorsFormatForField_whenLuceneFlat_thenReturnFlatFormat() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, flatMethodContext);

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.FLAT,
            ctx -> FLAT_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame(FLAT_FORMAT, result);
    }

    /**
     * When the Lucene engine is used with an SQ encoder, the SCALAR_QUANTIZED
     * resolver should be called.
     */
    public void testGetKnnVectorsFormatForField_whenLuceneSQ_thenReturnSQFormat() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNMethodContext sqMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, params)
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, sqMethodContext);

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.SCALAR_QUANTIZED,
            ctx -> SQ_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame(SQ_FORMAT, result);
    }

    /**
     * When the Lucene engine is used with a Optimized Scalar Quantizer encoder
     * resolver should be called.
     */
    public void testGetKnnVectorsFormatForField_whenLuceneSQ_thenReturnLuceneSQFormat() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNMethodContext OptimizedScalarQuantizerMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, params)
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, OptimizedScalarQuantizerMethodContext);

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.OPTIMIZED_SCALAR_QUANTIZER,
            ctx -> OPTIMIZED_SCALAR_QUANTIZER_FORMAT,
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame(OPTIMIZED_SCALAR_QUANTIZER_FORMAT, result);
    }

    /**
     * When the Lucene engine requests a format type that is not registered, an
     * exception should be thrown.
     */
    public void testGetKnnVectorsFormatForField_whenFormatNotRegistered_thenThrowException() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, flatMethodContext);

        // Register only HNSW, but flat method will request FLAT
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        expectThrows(IllegalStateException.class, () -> format.getKnnVectorsFormatForField(TEST_FIELD));
    }

    /**
     * Verify that the context passed to resolvers contains the correct field name
     * and params.
     */
    public void testResolveLuceneFormat_contextContainsCorrectValues() {
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

        MapperService mapperService = mockMapperService(TEST_FIELD, methodContext);

        // Capture the context via the resolver
        final KnnVectorsFormatContext[] capturedContext = new KnnVectorsFormatContext[1];
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> {
                capturedContext[0] = ctx;
                return HNSW_FORMAT;
            }
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        format.getKnnVectorsFormatForField(TEST_FIELD);

        assertNotNull(capturedContext[0]);
        assertEquals(TEST_FIELD, capturedContext[0].getField());
        assertEquals(DEFAULT_MAX_CONN, capturedContext[0].getDefaultMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, capturedContext[0].getDefaultBeamWidth());
        assertSame(methodContext, capturedContext[0].getMethodContext());
        assertSame(params, capturedContext[0].getParams());
    }

    /**
     * When the field has a model ID, the native engine format should be returned.
     */
    public void testGetKnnVectorsFormatForField_whenModelIdPresent_thenReturnNativeFormat() {
        MapperService mapperService = mockMapperServiceWithModelId(TEST_FIELD, "test-model-id");

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), Map.of());
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }

    /**
     * When the engine is not Lucene (e.g., FAISS), the native engine format should
     * be returned.
     */
    public void testGetKnnVectorsFormatForField_whenNativeEngine_thenReturnNativeFormat() {
        KNNMethodContext faissMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, faissMethodContext);

        // Even though we register Lucene resolvers, FAISS should bypass them entirely
        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            ctx -> HNSW_FORMAT
        );

        TestPerFieldKnnVectorsFormat format = new TestPerFieldKnnVectorsFormat(Optional.of(mapperService), resolvers);
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }

    /**
     * When a legacy-style format (with vectorsFormatSupplier) resolves a field
     * with HNSW parameters and no encoder, the unified routing path should
     * determine HNSW format type and delegate to the vectorsFormatSupplier.
     */
    public void testGetKnnVectorsFormatForField_legacyHnswWithoutEncoder_thenReturnHnswFormat() {
        KNNMethodContext hnswMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 32, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, hnswMethodContext);

        // Capture the params passed to the legacy supplier to verify correct routing
        final KNNVectorsFormatParams[] capturedParams = new KNNVectorsFormatParams[1];
        Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsFormatSupplier = params -> {
            capturedParams[0] = params;
            return HNSW_FORMAT;
        };

        LegacyTestPerFieldKnnVectorsFormat format = new LegacyTestPerFieldKnnVectorsFormat(
            Optional.of(mapperService),
            vectorsFormatSupplier
        );
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);

        assertSame(HNSW_FORMAT, result);
        assertNotNull(capturedParams[0]);
        assertEquals(32, capturedParams[0].getMaxConnections());
        assertEquals(256, capturedParams[0].getBeamWidth());
    }

    /**
     * When a legacy-style format (with both vectorsFormatSupplier and
     * scalarQuantizedVectorsFormatSupplier) resolves a field with SQ encoder
     * parameters, the unified routing path should determine SCALAR_QUANTIZED
     * format type and delegate to the scalarQuantizedVectorsFormatSupplier.
     */
    public void testGetKnnVectorsFormatForField_legacySQWithEncoder_thenReturnSQFormat() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNMethodContext sqMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, params)
        );

        MapperService mapperService = mockMapperService(TEST_FIELD, sqMethodContext);

        // Capture the params passed to the legacy SQ supplier to verify correct routing
        final KNNScalarQuantizedVectorsFormatParams[] capturedSQParams = new KNNScalarQuantizedVectorsFormatParams[1];
        Function<KNNScalarQuantizedVectorsFormatParams, KnnVectorsFormat> sqSupplier = sqParams -> {
            capturedSQParams[0] = sqParams;
            return SQ_FORMAT;
        };

        // The HNSW supplier should NOT be called for SQ routing
        Function<KNNVectorsFormatParams, KnnVectorsFormat> hnswSupplier = hnswParams -> {
            fail("HNSW supplier should not be called when SQ encoder is present");
            return HNSW_FORMAT;
        };

        LegacySQTestPerFieldKnnVectorsFormat format = new LegacySQTestPerFieldKnnVectorsFormat(
            Optional.of(mapperService),
            hnswSupplier,
            sqSupplier
        );
        KnnVectorsFormat result = format.getKnnVectorsFormatForField(TEST_FIELD);

        assertSame(SQ_FORMAT, result);
        assertNotNull(capturedSQParams[0]);
        assertEquals(16, capturedSQParams[0].getMaxConnections());
        assertEquals(100, capturedSQParams[0].getBeamWidth());
    }

    /**
     *
     * Parameterized property test verifying that non-Lucene fields (non-KNN, model-based, native engine)
     * are correctly routed by the legacy BasePerFieldKnnVectorsFormat.
     *
     */
    public void testLegacyRoutingCorrectness_nonLuceneFields() {
        Function<KNNVectorsFormatParams, KnnVectorsFormat> vectorsSupplier = params -> {
            fail("vectorsFormatSupplier should not be called for non-Lucene fields");
            return HNSW_FORMAT;
        };

        // 1. Non-KNN field → default format
        LegacyTestPerFieldKnnVectorsFormat defaultFormat = new LegacyTestPerFieldKnnVectorsFormat(Optional.empty(), vectorsSupplier);
        KnnVectorsFormat result = defaultFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame("Non-KNN field should return default format", DEFAULT_FORMAT, result);

        // 2. Model-based field → NativeEngines990KnnVectorsFormat
        MapperService modelMapper = mockMapperServiceWithModelId(TEST_FIELD, "test-model-1");
        LegacyTestPerFieldKnnVectorsFormat modelFormat = new LegacyTestPerFieldKnnVectorsFormat(Optional.of(modelMapper), vectorsSupplier);
        result = modelFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue("Model-based field should return NativeEngines990KnnVectorsFormat", result instanceof NativeEngines990KnnVectorsFormat);

        // 3. Another model-based field with different model ID
        MapperService modelMapper2 = mockMapperServiceWithModelId(TEST_FIELD, "another-model-id");
        LegacyTestPerFieldKnnVectorsFormat modelFormat2 = new LegacyTestPerFieldKnnVectorsFormat(
            Optional.of(modelMapper2),
            vectorsSupplier
        );
        result = modelFormat2.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(
            "Model-based field (different model) should return NativeEngines990KnnVectorsFormat",
            result instanceof NativeEngines990KnnVectorsFormat
        );

        // 4. FAISS engine → NativeEngines990KnnVectorsFormat
        KNNMethodContext faissContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );
        MapperService faissMapper = mockMapperService(TEST_FIELD, faissContext);
        LegacyTestPerFieldKnnVectorsFormat faissFormat = new LegacyTestPerFieldKnnVectorsFormat(Optional.of(faissMapper), vectorsSupplier);
        result = faissFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue("FAISS engine should return NativeEngines990KnnVectorsFormat", result instanceof NativeEngines990KnnVectorsFormat);

        // 5. FAISS engine with different params
        KNNMethodContext faissContext2 = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 32, METHOD_PARAMETER_EF_CONSTRUCTION, 512))
        );
        MapperService faissMapper2 = mockMapperService(TEST_FIELD, faissContext2);
        LegacyTestPerFieldKnnVectorsFormat faissFormat2 = new LegacyTestPerFieldKnnVectorsFormat(
            Optional.of(faissMapper2),
            vectorsSupplier
        );
        result = faissFormat2.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(
            "FAISS engine (different params) should return NativeEngines990KnnVectorsFormat",
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }

    /**
     *
     * Verifies that non-Lucene fields (non-KNN, model-based, native engine) bypass the registry
     * entirely in KNN1040BasePerFieldKnnVectorsFormat.
     *
     */
    public void testRegistryRoutingCorrectness_nonLuceneFields() {
        Function<KnnVectorsFormatContext, KnnVectorsFormat> failResolver = ctx -> {
            fail("Registry resolver should not be called for non-Lucene fields");
            return HNSW_FORMAT;
        };

        Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers = Map.of(
            LuceneVectorsFormatType.HNSW,
            failResolver,
            LuceneVectorsFormatType.SCALAR_QUANTIZED,
            failResolver,
            LuceneVectorsFormatType.FLAT,
            failResolver
        );

        // 1. Non-KNN field → default format
        TestPerFieldKnnVectorsFormat defaultFormat = new TestPerFieldKnnVectorsFormat(Optional.empty(), resolvers);
        KnnVectorsFormat result = defaultFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertSame("Non-KNN field should return default format", DEFAULT_FORMAT, result);

        // 2. Model-based field → NativeEngines990KnnVectorsFormat
        MapperService modelMapper = mockMapperServiceWithModelId(TEST_FIELD, "test-model-registry");
        TestPerFieldKnnVectorsFormat modelFormat = new TestPerFieldKnnVectorsFormat(Optional.of(modelMapper), resolvers);
        result = modelFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue("Model-based field should return NativeEngines990KnnVectorsFormat", result instanceof NativeEngines990KnnVectorsFormat);

        // 3. FAISS engine → NativeEngines990KnnVectorsFormat
        KNNMethodContext faissContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_M, 16, METHOD_PARAMETER_EF_CONSTRUCTION, 256))
        );
        MapperService faissMapper = mockMapperService(TEST_FIELD, faissContext);
        TestPerFieldKnnVectorsFormat faissFormat = new TestPerFieldKnnVectorsFormat(Optional.of(faissMapper), resolvers);
        result = faissFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue("FAISS engine should return NativeEngines990KnnVectorsFormat", result instanceof NativeEngines990KnnVectorsFormat);
    }

    /**
     * Helper to create a KNNMethodContext with SQ encoder parameters for the legacy routing tests.
     */
    private static KNNMethodContext createSQMethodContext(int m, int efConstruction, Map<String, Object> extraEncoderParams) {
        Map<String, Object> encoderParams = new HashMap<>(extraEncoderParams);
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, m);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);

        return new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, params));
    }

    /**
     * Helper to create a mocked MapperService that returns a KNNVectorFieldType
     * with the given method context.
     */
    private MapperService mockMapperService(String fieldName, KNNMethodContext knnMethodContext) {
        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            fieldName,
            Collections.emptyMap(),
            org.opensearch.knn.index.VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        when(mapperService.fieldType(eq(fieldName))).thenReturn(fieldType);

        // Mock IndexSettings for the approximate threshold
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        return mapperService;
    }

    /**
     * Helper to create a mocked MapperService that returns a KNNVectorFieldType
     * with a model ID (triggers native engine format).
     */
    private MapperService mockMapperServiceWithModelId(String fieldName, String modelId) {
        MapperService mapperService = mock(MapperService.class);

        KNNMappingConfig modelMappingConfig = new KNNMappingConfig() {
            @Override
            public Optional<String> getModelId() {
                return Optional.of(modelId);
            }

            @Override
            public int getDimension() {
                return 3;
            }
        };

        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            fieldName,
            Collections.emptyMap(),
            org.opensearch.knn.index.VectorDataType.FLOAT,
            modelMappingConfig
        );
        when(mapperService.fieldType(eq(fieldName))).thenReturn(fieldType);

        // Mock IndexSettings for the approximate threshold
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        return mapperService;
    }

    // --- SQ with bits=1 (1-bit quantization) format routing ---

    public void testGetKnnVectorsFormatForField_whenSQOneBit_thenReturnSQOneBitFormat() {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        MethodComponentContext hnswContext = new MethodComponentContext("hnsw", Map.of(METHOD_ENCODER_PARAMETER, encoderContext));
        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(methodContext.getMethodComponentContext()).thenReturn(hnswContext);

        MapperService mapperService = mockMapperService(TEST_FIELD, methodContext);
        KNN1040PerFieldKnnVectorsFormat perFieldFormat = new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService));
        KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(format instanceof Faiss1040ScalarQuantizedKnnVectorsFormat);
    }

    public void testGetKnnVectorsFormatForField_whenSQWithoutOneBit_thenReturnNativeFormat() {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        MethodComponentContext hnswContext = new MethodComponentContext("hnsw", Map.of(METHOD_ENCODER_PARAMETER, encoderContext));
        KNNMethodContext methodContext = mock(KNNMethodContext.class);
        when(methodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(methodContext.getMethodComponentContext()).thenReturn(hnswContext);

        MapperService mapperService = mockMapperService(TEST_FIELD, methodContext);
        KNN1040PerFieldKnnVectorsFormat perFieldFormat = new KNN1040PerFieldKnnVectorsFormat(Optional.of(mapperService));
        KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(TEST_FIELD);
        assertTrue(format instanceof NativeEngines990KnnVectorsFormat);
    }
}
