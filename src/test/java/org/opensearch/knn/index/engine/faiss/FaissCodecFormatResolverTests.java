/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;
import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Unit tests for {@link FaissCodecFormatResolver}.
 * Validates approximateThreshold resolution from index settings and
 * correct pass-through of NativeIndexBuildStrategyFactory.
 */
public class FaissCodecFormatResolverTests extends KNNTestCase {

    private static final String TEST_FIELD = "test_vector";
    private static final int DEFAULT_MAX_CONN = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    public void testResolve_whenCustomApproximateThreshold_thenFormatUsesCustomValue() {
        int customThreshold = 5000;

        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(customThreshold);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        NativeIndexBuildStrategyFactory factory = mock(NativeIndexBuildStrategyFactory.class);

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(Optional.of(mapperService), factory);
        KnnVectorsFormat result = resolver.resolve();

        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
        assertTrue(
            "Format should contain custom threshold value " + customThreshold + ", but was: " + result.toString(),
            result.toString().contains("approximateThreshold=" + customThreshold)
        );
    }

    public void testResolve_whenApproximateThresholdIsNull_thenFormatUsesDefaultValue() {
        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        NativeIndexBuildStrategyFactory factory = mock(NativeIndexBuildStrategyFactory.class);

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(Optional.of(mapperService), factory);
        KnnVectorsFormat result = resolver.resolve();

        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
        int defaultThreshold = KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
        assertTrue(
            "Format should contain default threshold value " + defaultThreshold + ", but was: " + result.toString(),
            result.toString().contains("approximateThreshold=" + defaultThreshold)
        );
    }

    public void testResolve_whenNativeIndexBuildStrategyFactoryProvided_thenPassedThroughToFormat() {
        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        NativeIndexBuildStrategyFactory factory = new NativeIndexBuildStrategyFactory();

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(Optional.of(mapperService), factory);
        KnnVectorsFormat result = resolver.resolve();

        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }

    public void testResolve_whenCalledWithFieldContext_andBBQEncoder_thenReturnsBBQFormat() {
        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(
            Optional.of(mapperService),
            mock(NativeIndexBuildStrategyFactory.class)
        );

        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1));
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderContext);

        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, null, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertTrue(
            "Expected Faiss104ScalarQuantizedKnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof Faiss1040ScalarQuantizedKnnVectorsFormat
        );
    }

    public void testResolve_whenCalledWithFieldContext_andNoEncoder_thenReturnsNativeFormat() {
        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(
            Optional.of(mapperService),
            mock(NativeIndexBuildStrategyFactory.class)
        );

        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, null, Map.of(), DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }

    public void testResolve_whenCalledWithFieldContext_andNonBBQEncoder_thenReturnsNativeFormat() {
        MapperService mapperService = mock(MapperService.class);
        IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(null);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        FaissCodecFormatResolver resolver = new FaissCodecFormatResolver(
            Optional.of(mapperService),
            mock(NativeIndexBuildStrategyFactory.class)
        );

        MethodComponentContext encoderContext = new MethodComponentContext("sq", Map.of());
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderContext);

        KnnVectorsFormat result = resolver.resolve(TEST_FIELD, null, params, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof NativeEngines990KnnVectorsFormat
        );
    }
}
