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
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.KNNVectorDocValueFormat;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
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
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_FLAT)
            .encoderType(Encoder.EncoderType.FLAT)
            .compressionLevel(CompressionLevel.x32)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT, spec);
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
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT, spec);
    }

    public void testKNNVectorFieldType_resolvedSpecStoredWhenProvided() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNMethodContext sqOneBitMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of("bits", 1)))
            )
        );
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(sqOneBitMethodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig,
            Version.CURRENT,
            spec
        );
        assertNotNull(fieldType.getResolvedSpec());
        assertSame(spec, fieldType.getResolvedSpec());
        assertEquals(Encoder.EncoderType.SQ, fieldType.getResolvedSpec().getEncoderType());
    }

    public void testKNNVectorFieldType_noAnnSpecWhenNotProvided() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(knnMethodContext, 128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig,
            Version.CURRENT
        );
        // When no spec is provided, the field type defaults to a no-ANN spec with all behavior off
        ResolvedIndexSpec spec = fieldType.getResolvedSpec();
        assertNotNull(spec);
        assertNull(spec.getEngine());
        assertEquals(VectorDataType.FLOAT, spec.getVectorDataType());
        assertEquals(128, spec.getDimension());
        assertFalse(spec.supportsRadialSearch());
        assertFalse(spec.supportsRemoteIndexBuild());
        assertFalse(spec.alwaysUseMemoryOptimizedSearch());
        assertFalse(spec.isMemoryOptimizedEligible());
        assertNull(spec.getRescoreContext());
    }

    public void testKNNVectorFieldType_lazySpecSupplierMemoized() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNMappingConfig mappingConfig = getMappingConfigForMethodMapping(knnMethodContext, 128);
        final int[] calls = { 0 };
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig,
            Version.CURRENT,
            () -> {
                calls[0]++;
                return ResolvedIndexSpec.noAnn(VectorDataType.FLOAT, 128, Version.CURRENT);
            }
        );
        assertEquals(0, calls[0]);
        ResolvedIndexSpec first = fieldType.getResolvedSpec();
        ResolvedIndexSpec second = fieldType.getResolvedSpec();
        assertSame(first, second);
        assertEquals(1, calls[0]);
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
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig,
            Version.CURRENT,
            spec
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

    // --- radial search support via resolved spec ---

    public void testRadialSearchSupport_whenNoAnnSpec_thenNotSupported() {
        // Fields without an ANN structure (flat mapper, pre-method-serialization models) get a no-ANN spec
        KNNMappingConfig config = getMappingConfigForFlatMapping(128);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, config);
        assertFalse(fieldType.getResolvedSpec().supportsRadialSearch());
    }

    public void testRadialSearchSupport_whenSQOneBit_thenNotSupported() {
        // SQ 1-bit is a quantized index, so radial search is blocked (#3464) — no exception for 1-bit SQ
        assertFalse(buildSQOneBitFieldType().getResolvedSpec().supportsRadialSearch());
    }

    public void testRadialSearchSupport_whenModelBasedQuantized_thenSupported() {
        // Model-derived specs skip the compression-level restriction (parity with the legacy model-path
        // validation, which only blocked BQ via QuantizationConfig)
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.PQ)
            .compressionLevel(CompressionLevel.x8)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .modelBased(true)
            .build();
        assertTrue(spec.supportsRadialSearch());
        // The same configuration on a method-mapped field is blocked
        assertFalse(spec.toBuilder().modelBased(false).build().supportsRadialSearch());
    }

    public void testRadialSearchSupport_whenModelBasedBQ_thenNotSupported() {
        // BQ remains blocked even for model-derived specs
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .modelBased(true)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }
}
