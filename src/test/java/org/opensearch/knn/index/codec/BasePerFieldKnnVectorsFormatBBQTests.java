/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for Faiss BBQ routing through KNN1040BasePerFieldKnnVectorsFormat / FaissCodecFormatResolver.
 */
public class BasePerFieldKnnVectorsFormatBBQTests extends KNNTestCase {

    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissBBQEncoder_thenReturnsFaissBBQFormat() {
        final String fieldName = "bbq_field";
        final KNN1040PerFieldKnnVectorsFormat perFieldFormat = new KNN1040PerFieldKnnVectorsFormat(
            Optional.of(mockMapperService(fieldName, ENCODER_FAISS_BBQ))
        );
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);
        assertTrue(
            "Expected Faiss104ScalarQuantizedKnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof Faiss1040ScalarQuantizedKnnVectorsFormat
        );
    }

    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissWithoutBBQ_thenReturnsNativeFormat() {
        final String fieldName = "regular_faiss_field";
        final KNN1040PerFieldKnnVectorsFormat perFieldFormat = new KNN1040PerFieldKnnVectorsFormat(
            Optional.of(mockMapperService(fieldName, null))
        );
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof NativeEngines990KnnVectorsFormat
        );
    }

    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissWithNonBBQEncoder_thenReturnsNativeFormat() {
        final String fieldName = "non_bbq_field";
        final KNN1040PerFieldKnnVectorsFormat perFieldFormat = new KNN1040PerFieldKnnVectorsFormat(
            Optional.of(mockMapperService(fieldName, "some_other_encoder"))
        );
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);
        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof NativeEngines990KnnVectorsFormat
        );
    }

    /**
     * Creates a mock MapperService that returns a KNNVectorFieldType with Faiss engine
     * and optionally a specific encoder name.
     */
    private MapperService mockMapperService(String fieldName, String encoderName) {
        final Map<String, Object> params;
        if (encoderName != null) {
            final MethodComponentContext encoderContext = new MethodComponentContext(encoderName, Map.of("bits", 1));
            params = Map.of(METHOD_ENCODER_PARAMETER, encoderContext);
        } else {
            params = Map.of();
        }
        final MethodComponentContext methodComponentContext = new MethodComponentContext("hnsw", params);

        final KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);

        final KNNMappingConfig knnMappingConfig = mock(KNNMappingConfig.class);
        when(knnMappingConfig.getModelId()).thenReturn(Optional.empty());
        when(knnMappingConfig.getKnnMethodContext()).thenReturn(Optional.of(knnMethodContext));
        when(knnMappingConfig.getKnnLibraryIndexingContext()).thenReturn(null);

        final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(fieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);

        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(fieldName)).thenReturn(fieldType);

        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING)).thenReturn(
            KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE
        );
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        return mapperService;
    }
}
