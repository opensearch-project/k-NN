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
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

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

    // --- validateSupportRadialSearch tests ---

    public void testValidateRadialSearch_whenUnsupportedEngine_thenThrows() {
        // Given: a field type with NMSLIB engine (not in ENGINES_SUPPORTING_RADIAL_SEARCH)
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.NMSLIB,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = getMappingConfigForMethodMapping(methodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: throws with engine name
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> fieldType.validateSupportRadialSearch(KNNEngine.NMSLIB)
        );
        assertTrue(e.getMessage().contains("NMSLIB"));
    }

    public void testValidateRadialSearch_whenBinaryDataType_thenThrows() {
        // Given: a field type with BINARY data type
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.HAMMING,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = getMappingConfigForMethodMapping(methodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.BINARY, config);

        // When/Then: throws with binary message
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> fieldType.validateSupportRadialSearch(KNNEngine.FAISS)
        );
        assertTrue(e.getMessage().contains("Binary"));
    }

    public void testValidateRadialSearch_whenBQQuantized_thenThrows() {
        // Given: a field type with BQ quantization (QuantizationConfig != EMPTY)
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
            }

            @Override
            public int getDimension() {
                return 128;
            }

            @Override
            public QuantizationConfig getQuantizationConfig() {
                return QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
            }
        };
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: throws with binary quantization message
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> fieldType.validateSupportRadialSearch(KNNEngine.FAISS)
        );
        assertTrue(e.getMessage().contains("binary quantization"));
    }

    public void testValidateRadialSearch_whenUnsupportedCompressionLevel_thenThrows() {
        // Given: a field type with x8 compression and non-SQ-1-bit encoder
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
            }

            @Override
            public int getDimension() {
                return 128;
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                return CompressionLevel.x8;
            }
        };
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: throws with compression level in message
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> fieldType.validateSupportRadialSearch(KNNEngine.FAISS)
        );
        assertTrue(e.getMessage().contains("compression level=x8"));
    }

    public void testValidateRadialSearch_whenFlatMethod32x_thenPasses() {
        // Given: a flat method field type with 32x compression
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );
        KNNMappingConfig config = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
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
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: no exception — flat method with 32x is recognized as 1-bit SQ
        fieldType.validateSupportRadialSearch(KNNEngine.LUCENE);
    }

    public void testValidateRadialSearch_whenSQOneBit_thenPasses() {
        // Given: a field type with SQ encoder bits=1
        KNNVectorFieldType fieldType = buildSQOneBitFieldType();

        // When/Then: no exception — SQ 1-bit is supported via rescoring
        fieldType.validateSupportRadialSearch(KNNEngine.FAISS);
    }

    public void testValidateRadialSearch_whenNonQuantized_thenPasses() {
        // Given: a non-quantized field type (NOT_CONFIGURED compression)
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = getMappingConfigForMethodMapping(methodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: no exception — non-quantized indices always support radial search
        fieldType.validateSupportRadialSearch(KNNEngine.FAISS);
    }

    public void testValidateRadialSearch_whenX32HnswNonSQEncoder_thenThrows() {
        // Given: x32 compression with HNSW method and flat encoder (NOT SQ 1-bit, NOT flat method)
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
            )
        );
        KNNMappingConfig config = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
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
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: throws — x32 with HNSW and non-SQ encoder is not allowed
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> fieldType.validateSupportRadialSearch(KNNEngine.FAISS)
        );
        assertTrue(e.getMessage().contains("1-bit SQ"));
        assertTrue(e.getMessage().contains("x32"));
    }

    public void testValidateRadialSearch_whenFp16Compression_thenPasses() {
        // Given: a field type with x2 (fp16) compression — not quantized
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
            }

            @Override
            public int getDimension() {
                return 128;
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                return CompressionLevel.x2;
            }
        };
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: no exception — fp16 (x2) is not quantized, passes
        fieldType.validateSupportRadialSearch(KNNEngine.FAISS);
    }

    public void testValidateRadialSearch_whenNoMethodContext_thenPasses() {
        // Given: a model-based field type with no method context
        KNNMappingConfig config = getMappingConfigForFlatMapping(128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: no exception — model-based indices skip quantization check
        fieldType.validateSupportRadialSearch(KNNEngine.FAISS);
    }

    // --- isRescoringRequiredForRadial tests ---

    public void testIsRescoringRequired_whenSQOneBit_thenTrue() {
        // Given: a field type with SQ 1-bit encoder
        KNNVectorFieldType fieldType = buildSQOneBitFieldType();

        // When/Then: rescoring is required
        assertTrue(fieldType.isRescoringRequiredForRadial());
    }

    public void testIsRescoringRequired_whenNonQuantized_thenFalse() {
        // Given: a non-quantized field type
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        KNNMappingConfig config = getMappingConfigForMethodMapping(methodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: rescoring is not required
        assertFalse(fieldType.isRescoringRequiredForRadial());
    }

    public void testIsRescoringRequired_whenNoMethodContext_thenFalse() {
        // Given: a model-based field type with no method context
        KNNMappingConfig config = getMappingConfigForFlatMapping(128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);

        // When/Then: rescoring is not required (no method context to check)
        assertFalse(fieldType.isRescoringRequiredForRadial());
    }

    public void testIsRescoringRequired_whenFlatEncoder_thenFalse() {
        // Given: a field type with flat encoder (not SQ 1-bit)
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
            )
        );
        KNNMappingConfig config = getMappingConfigForMethodMapping(methodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            config,
            Version.CURRENT
        );

        // When/Then: rescoring is not required — flat encoder is not SQ 1-bit
        assertFalse(fieldType.isRescoringRequiredForRadial());
    }
}
