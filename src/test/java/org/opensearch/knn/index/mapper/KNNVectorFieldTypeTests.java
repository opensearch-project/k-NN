/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class KNNVectorFieldTypeTests extends KNNTestCase {
    private static final String FIELD_NAME = "test-field";

    public void testValueFetcher() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        ValueFetcher valueFetcher = knnVectorFieldType.valueFetcher(mockQueryShardContext, null, null);
        assertTrue(valueFetcher instanceof ArraySourceValueFetcher);
    }

    public void testResolveRescoreContext_whenFlatMethod_thenReturnOversampleFactor2() {
        RescoreContext rescoreContext = buildFlatFieldType().resolveRescoreContext(null);
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isUserProvided());
    }

    public void testResolveRescoreContext_whenFlatMethodWithUserProvidedContext_thenReturnUserContext() {
        RescoreContext userContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();
        assertSame(userContext, buildFlatFieldType().resolveRescoreContext(userContext));
    }

    // After resolution, flat method always has x32 compression set in the mapping config
    private KNNVectorFieldType buildFlatFieldType() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        KNNMappingConfig mappingConfig = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(flatMethodContext);
            }

            @Override
            public int getDimension() {
                return 128;
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                return CompressionLevel.x32;
            }
        };
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig);
    }

    public void testKNNVectorFieldType_whenBBQEncoder_thenAlwaysUseMemoryOptimizedSearchIsTrue() {
        KNNVectorFieldType fieldType = buildBBQFieldType();
        assertTrue(fieldType.isAlwaysUseMemoryOptimizedSearch());
        assertTrue(fieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testResolveRescoreContext_whenBBQEncoder_thenReturnFixedOversampleFactor() {
        KNNVectorFieldType fieldType = buildBBQFieldType();
        RescoreContext rescoreContext = fieldType.resolveRescoreContext(null);
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isUserProvided());
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
        assertTrue(rescoreContext.isRescoreEnabled());
    }

    public void testResolveRescoreContext_whenBBQEncoderWithUserProvidedContext_thenReturnUserContext() {
        RescoreContext userContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();
        assertSame(userContext, buildBBQFieldType().resolveRescoreContext(userContext));
    }

    public void testResolveRescoreContext_whenNoMethodContext_thenReturnsNull() {
        KNNMappingConfig mappingConfig = getMappingConfigForFlatMapping(128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig);
        assertNull(fieldType.resolveRescoreContext(null));
    }

    private KNNVectorFieldType buildBBQFieldType() {
        KNNMethodContext bbqMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FAISS_BBQ, Collections.emptyMap()))
            )
        );
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(bbqMethodContext, 128);
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT);
    }

    public void testKNNVectorFieldType_whenNonBBQEncoder_thenAlwaysUseMemoryOptimizedSearchIsFalse() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
            )
        );
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(flatMethodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig,
            Version.CURRENT
        );
        assertFalse(fieldType.isAlwaysUseMemoryOptimizedSearch());
        assertTrue(fieldType.isMemoryOptimizedSearchAvailable());
    }
}
