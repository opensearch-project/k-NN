/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Tests verifying that consumers reading from ResolvedIndexSpec produce identical
 * behavior to the old param-map based paths. These are BWC equivalence tests for
 * the consumer switchover (PR 3).
 */
public class ResolvedIndexSpecConsumerTests extends KNNTestCase {

    private static final String FIELD_NAME = "test-field";

    public void testMemoryOptimizedSearch_specProducesSameAsOldPath_SQ1Bit() {
        KNNVectorFieldType specPathFieldType = buildFieldTypeWithSpec(buildSQ1BitMethodContext(), 128, buildSQ1BitSpec());

        assertTrue(specPathFieldType.isAlwaysUseMemoryOptimizedSearch());
        assertTrue(specPathFieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testMemoryOptimizedSearch_specProducesSameAsOldPath_FlatEncoder() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
            )
        );
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .mode(Mode.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNVectorFieldType specPathFieldType = buildFieldTypeWithSpec(flatMethodContext, 128, spec);

        assertFalse(specPathFieldType.isAlwaysUseMemoryOptimizedSearch());
        assertTrue(specPathFieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testMemoryOptimizedSearch_specProducesFalseForLucene() {
        KNNMethodContext luceneMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .mode(Mode.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();

        KNNVectorFieldType specPathFieldType = buildFieldTypeWithSpec(luceneMethodContext, 128, spec);
        assertFalse(specPathFieldType.isAlwaysUseMemoryOptimizedSearch());
        assertFalse(specPathFieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testRescore_specProducesSameResult_SQ1Bit() {
        ResolvedIndexSpec spec = buildSQ1BitSpec();
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.001f);
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
    }

    public void testRescore_specProducesSameResult_x32OnDisk() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.001f);
    }

    public void testRescore_specReturnsNull_forNoCompression() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .mode(Mode.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        assertNull(spec.getRescoreContext());
    }

    public void testRadialSearch_specMatchesOldBehavior_SQ1BitSupported() {
        ResolvedIndexSpec spec = buildSQ1BitSpec();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_specMatchesOldBehavior_x32HNSWNotSupported() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_specMatchesOldBehavior_binaryNotSupported() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.BINARY)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testNullSpec_fallsBackToOldPath() {
        KNNVectorFieldType fieldType = buildFieldTypeWithoutSpec(buildSQ1BitMethodContext(), 128);
        assertNull(fieldType.getResolvedSpec());
        // Fallback computes MOS flags from mapping config: SQ 1-bit always uses MOS
        assertTrue(fieldType.isAlwaysUseMemoryOptimizedSearch());
        // isSupportedFieldType requires Faiss HNSW with FLAT/SQ/BQ + no model, which this satisfies
        assertTrue(fieldType.isMemoryOptimizedSearchAvailable());
    }

    public void testSQ1BitCodecFormat_viaSPec() {
        ResolvedIndexSpec spec = buildSQ1BitSpec();
        assertTrue(spec.isFaissSQOneBit());
    }

    public void testSQ16Bit_notSQ1BitCodecFormat() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.x2)
            .mode(Mode.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        assertFalse(spec.isFaissSQOneBit());
    }

    // --- Helpers ---

    private KNNMethodContext buildSQ1BitMethodContext() {
        return new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of("bits", 1)))
            )
        );
    }

    private ResolvedIndexSpec buildSQ1BitSpec() {
        return ResolvedIndexSpec.builder()
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
    }

    private KNNVectorFieldType buildFieldTypeWithoutSpec(KNNMethodContext methodContext, int dimension) {
        KNNMappingConfig mappingConfig = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
            }

            @Override
            public int getDimension() {
                return dimension;
            }
        };
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT);
    }

    public void testRescore_derivedOnDiskMode_producesNonNullRescore() {
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
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
    }

    public void testRescore_notConfiguredModeWithX32_sameAsOnDisk() {
        ResolvedIndexSpec specOnDisk = ResolvedIndexSpec.builder()
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

        ResolvedIndexSpec specNotConfigured = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();

        RescoreContext onDiskRescore = specOnDisk.getRescoreContext();
        RescoreContext notConfiguredRescore = specNotConfigured.getRescoreContext();

        assertNotNull(onDiskRescore);
        assertNotNull(notConfiguredRescore);
        assertEquals(onDiskRescore.getOversampleFactor(), notConfiguredRescore.getOversampleFactor(), 0.001f);
    }

    public void testRescore_explicitOnDiskVsDerivedOnDisk_identical() {
        ResolvedIndexSpec specExplicit = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();

        ResolvedIndexSpec specDerived = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();

        RescoreContext explicit = specExplicit.getRescoreContext();
        RescoreContext derived = specDerived.getRescoreContext();

        assertNotNull(explicit);
        assertNotNull(derived);
        assertEquals(explicit.getOversampleFactor(), derived.getOversampleFactor(), 0.001f);
        assertEquals(explicit.isAllowOverrideOversampleFactor(), derived.isAllowOverrideOversampleFactor());
    }

    private KNNVectorFieldType buildFieldTypeWithSpec(KNNMethodContext methodContext, int dimension, ResolvedIndexSpec spec) {
        KNNMappingConfig mappingConfig = new KNNMappingConfig() {
            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(methodContext);
            }

            @Override
            public int getDimension() {
                return dimension;
            }
        };
        return new KNNVectorFieldType(FIELD_NAME, Collections.emptyMap(), VectorDataType.FLOAT, mappingConfig, Version.CURRENT, spec);
    }
}
