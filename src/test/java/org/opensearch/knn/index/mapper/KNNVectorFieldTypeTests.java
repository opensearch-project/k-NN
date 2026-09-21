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
import org.opensearch.knn.index.KNNVectorDocValueFormat;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.search.DocValueFormat;

import java.time.ZoneId;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
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

    /**
     * half_float x16 resolves to SQ 1-bit, but the flat method carries no encoder in its method
     * context - the rescore default must still recognize it as SQ 1-bit, or the 1-bit first pass
     * runs without full-precision rescoring and recall@1 collapses (observed 0.55 vs 1.00).
     */
    public void testResolveRescoreContext_whenHalfFloatFlatX16_thenReturnFixedOversampleFactor() {
        RescoreContext rescoreContext = buildHalfFloatFlatFieldType(CompressionLevel.x16).resolveRescoreContext(null);
        assertNotNull("half_float flat x16 (SQ 1-bit) must get a default rescore context", rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isUserProvided());
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
    }

    /** half_float x1 is raw fp16 (exact scoring) - no default rescore applies. */
    public void testResolveRescoreContext_whenHalfFloatFlatX1_thenNull() {
        assertNull(buildHalfFloatFlatFieldType(CompressionLevel.x1).resolveRescoreContext(null));
    }

    private KNNVectorFieldType buildHalfFloatFlatFieldType(CompressionLevel compressionLevel) {
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
                return compressionLevel;
            }
        };
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.HALF_FLOAT, mappingConfig);
    }

    public void testKNNVectorFieldType_whenSQOneBitEncoder_thenAlwaysUseMemoryOptimizedSearchIsTrue() {
        KNNVectorFieldType fieldType = buildSQOneBitFieldType();
        assertTrue(fieldType.isAlwaysUseMemoryOptimizedSearch());
        assertTrue(fieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testResolveRescoreContext_whenSQOneBitEncoder_thenReturnFixedOversampleFactor() {
        KNNVectorFieldType fieldType = buildSQOneBitFieldType();
        RescoreContext rescoreContext = fieldType.resolveRescoreContext(null);
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isUserProvided());
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
        assertTrue(rescoreContext.isRescoreEnabled());
    }

    public void testResolveRescoreContext_whenSQOneBitEncoderWithUserProvidedContext_thenReturnUserContext() {
        RescoreContext userContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();
        assertSame(userContext, buildSQOneBitFieldType().resolveRescoreContext(userContext));
    }

    public void testResolveRescoreContext_whenNoMethodContext_thenReturnsNull() {
        KNNMappingConfig mappingConfig = getMappingConfigForFlatMapping(128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig);
        assertNull(fieldType.resolveRescoreContext(null));
    }

    private KNNVectorFieldType buildSQOneBitFieldType() {
        KNNMethodContext sqOneBitMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of("bits", 1)))
            )
        );
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(sqOneBitMethodContext, 128);
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT);
    }

    public void testKNNVectorFieldType_whenNonSQOneBitEncoder_thenAlwaysUseMemoryOptimizedSearchIsFalse() {
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

    public void testDocValueFormat_nullFormat_returnsBinaryFormat() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        DocValueFormat format = fieldType.docValueFormat(null, null);
        assertSame(KNNVectorDocValueFormat.BINARY_FORMAT, format);
    }

    public void testDocValueFormat_arrayFormat_returnsArrayFormat() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        DocValueFormat format = fieldType.docValueFormat("array", null);
        assertSame(KNNVectorDocValueFormat.ARRAY_FORMAT, format);
        assertFalse(((KNNVectorDocValueFormat) format).isBinary());
    }

    public void testDocValueFormat_binaryFormat_returnsBinaryFormat() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        DocValueFormat format = fieldType.docValueFormat("binary", null);
        assertSame(KNNVectorDocValueFormat.BINARY_FORMAT, format);
        assertTrue(((KNNVectorDocValueFormat) format).isBinary());
    }

    public void testDocValueFormat_unsupportedFormat_throwsIllegalArgument() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> fieldType.docValueFormat("epoch_millis", null));
        assertTrue(ex.getMessage().contains("epoch_millis"));
        assertTrue(ex.getMessage().contains("Unsupported knn_vector docvalue_fields format"));
    }

    public void testDocValueFormat_nonNullTimezone_throwsIllegalArgument() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, 3)
        );
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> fieldType.docValueFormat(null, ZoneId.of("UTC")));
        assertTrue(ex.getMessage().contains(FIELD_NAME));
        assertTrue(ex.getMessage().contains("does not support custom time zones"));
    }
}
