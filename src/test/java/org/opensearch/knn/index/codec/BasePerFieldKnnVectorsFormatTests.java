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
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static org.apache.lucene.tests.util.LuceneTestCase.expectThrows;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
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
    private static final KnnVectorsFormat FLAT_FORMAT = mock(KnnVectorsFormat.class);
    private static final KnnVectorsFormat DEFAULT_FORMAT = mock(KnnVectorsFormat.class);

    /**
     * Concrete subclass for testing the abstract base class.
     */
    private static class TestPerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {
        TestPerFieldKnnVectorsFormat(
            Optional<MapperService> mapperService,
            Map<LuceneVectorsFormatType, Function<KnnVectorsFormatContext, KnnVectorsFormat>> resolvers
        ) {
            super(
                mapperService,
                DEFAULT_MAX_CONN,
                DEFAULT_BEAM_WIDTH,
                () -> DEFAULT_FORMAT,
                resolvers,
                new NativeIndexBuildStrategyFactory()
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
}
