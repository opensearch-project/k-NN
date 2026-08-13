/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;

public class ResolvedIndexSpecTests extends KNNTestCase {

    // --- SQ 1-bit codec / memopt ---

    public void testFaissSQ1BitUsesSQ1BitCodecFormat() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.isFaissSQOneBit());
        assertTrue(spec.alwaysUseMemoryOptimizedSearch());
        assertTrue(spec.isMemoryOptimizedEligible());
        assertTrue(spec.requiresRescore());
    }

    public void testLuceneSQ1BitDoesNotUseSQ1BitCodecFormat() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().engine(KNNEngine.LUCENE).build();
        assertFalse(spec.isFaissSQOneBit());
        assertTrue(spec.alwaysUseMemoryOptimizedSearch());
    }

    // --- Memory optimized eligibility ---

    public void testFaissHNSWSQIsMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.isMemoryOptimizedEligible());
    }

    public void testFaissHNSWBQIsMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.TWO)
            .compressionLevel(CompressionLevel.x16)
            .build();
        assertTrue(spec.isMemoryOptimizedEligible());
    }

    public void testFaissIVFNotMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss().methodName(METHOD_IVF)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertFalse(spec.isMemoryOptimizedEligible());
    }

    public void testFaissPQNotMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.PQ)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.isMemoryOptimizedEligible());
    }

    // --- Radial search support ---

    public void testRadialSearch_NMSLIBNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().engine(KNNEngine.NMSLIB)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_BinaryDataTypeNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .vectorDataType(VectorDataType.BINARY)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_BQNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.TWO)
            .compressionLevel(CompressionLevel.x16)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_QuantizedSQ4BitNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_QuantizedPQNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.PQ)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_SQ1BitNotSupported() {
        // Radial search is now blocked for all quantized indices, including 1-bit SQ (#3464).
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_FlatMethodWithX32NotSupported() {
        // The former flat-method exception is gone: x32 is quantized, so radial search is blocked (#3464).
        ResolvedIndexSpec spec = baseFaiss().methodName(METHOD_FLAT)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x32)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_NonQuantizedAlwaysSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_LuceneNonQuantizedSupported() {
        ResolvedIndexSpec spec = baseFaiss().engine(KNNEngine.LUCENE)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SEVEN)
            .compressionLevel(CompressionLevel.x4)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_NotConfiguredCompressionSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertTrue(spec.supportsRadialSearch());
    }

    // --- Rescore context ---

    public void testRescoreContext_SQ1BitReturnsFixedOversample() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        RescoreContext expected = RescoreContext.builder()
            .oversampleFactor(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR)
            .allowOverrideOversampleFactor(false)
            .userProvided(false)
            .build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x1ReturnsNull() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertNull(spec.getRescoreContext());
        assertFalse(spec.requiresRescore());
    }

    public void testRescoreContext_x32OnDiskAboveThreshold() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .dimension(1500)
            .build();
        RescoreContext expected = RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x32OnDiskBelowThreshold() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .dimension(500)
            .build();
        RescoreContext expected = RescoreContext.builder()
            .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD)
            .userProvided(false)
            .build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x4OnDiskAboveThreshold() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SEVEN)
            .compressionLevel(CompressionLevel.x4)
            .mode(Mode.ON_DISK)
            .dimension(1500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        RescoreContext expected = RescoreContext.builder().oversampleFactor(1.0f).userProvided(false).build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x4BeforeV310ReturnsNull() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SEVEN)
            .compressionLevel(CompressionLevel.x4)
            .mode(Mode.ON_DISK)
            .dimension(1500)
            .indexVersionCreated(Version.V_2_19_0)
            .build();
        assertNull(spec.getRescoreContext());
    }

    public void testRescoreContext_x32FlatMethodReturnsFixedOversample() {
        ResolvedIndexSpec spec = baseFaiss().methodName(METHOD_FLAT)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .build();
        RescoreContext expected = RescoreContext.builder().oversampleFactor(2.0f).userProvided(false).build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x32LuceneAfterV360() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(768)
            .indexVersionCreated(Version.V_3_6_0)
            .build();
        // SQ 1-bit takes priority over Lucene-specific path
        RescoreContext expected = RescoreContext.builder()
            .oversampleFactor(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR)
            .allowOverrideOversampleFactor(false)
            .userProvided(false)
            .build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_x32LuceneAfterV360_NonSQ1Bit() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(1500)
            .indexVersionCreated(Version.V_3_6_0)
            .build();
        RescoreContext expected = RescoreContext.builder()
            .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_DEFAULT_FOR_LUCENE_SCALAR_QUANTIZER_AFTER_V360)
            .userProvided(false)
            .build();
        assertEquals(expected, spec.getRescoreContext());
    }

    public void testRescoreContext_NotOnDiskReturnsNull() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .dimension(500)
            .build();
        assertNull(spec.getRescoreContext());
    }

    // --- Memory optimized search for on-disk ---

    public void testRequiresMemoryOptimizedSearchForOnDisk_OnDiskX1() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .mode(Mode.ON_DISK)
            .build();
        assertTrue(spec.requiresMemoryOptimizedSearchForOnDisk());
    }

    public void testRequiresMemoryOptimizedSearchForOnDisk_OnDiskX32() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().mode(Mode.ON_DISK).build();
        assertFalse(spec.requiresMemoryOptimizedSearchForOnDisk());
    }

    public void testRequiresMemoryOptimizedSearchForOnDisk_InMemoryX1() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .mode(Mode.NOT_CONFIGURED)
            .build();
        assertFalse(spec.requiresMemoryOptimizedSearchForOnDisk());
    }

    // --- Builder defaults ---

    public void testBuilderDefaultsForCompressionAndMode() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT)
            .build();
        assertEquals(CompressionLevel.NOT_CONFIGURED, spec.getCompressionLevel());
        assertEquals(Mode.NOT_CONFIGURED, spec.getMode());
    }

    public void testInputsStoredCorrectly() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.LUCENE)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(256)
            .indexVersionCreated(Version.V_3_6_0)
            .build();
        assertEquals(KNNEngine.LUCENE, spec.getEngine());
        assertEquals(METHOD_HNSW, spec.getMethodName());
        assertEquals(Encoder.EncoderType.SQ, spec.getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, spec.getQuantizationBits());
        assertEquals(CompressionLevel.x32, spec.getCompressionLevel());
        assertEquals(Mode.ON_DISK, spec.getMode());
        assertEquals(VectorDataType.FLOAT, spec.getVectorDataType());
        assertEquals(256, spec.getDimension());
        assertEquals(Version.V_3_6_0, spec.getIndexVersionCreated());
    }

    // --- Additional branch coverage ---

    public void testFaissHNSWFlatIsMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertTrue(spec.isMemoryOptimizedEligible());
    }

    public void testGetRescoreContext_NullVersionDefaults() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x8)
            .dimension(500)
            .indexVersionCreated(null)
            .build();
        // Should not throw - null version defaults to CURRENT
        spec.getRescoreContext();
    }

    // --- Helpers ---

    // --- Coverage: alwaysUseMemoryOptimizedSearch engine-agnostic ---

    public void testAlwaysUseMemoryOptimizedSearch() {
        assertTrue("SQ 1-bit Faiss should always use memory optimized search", baseFaissSQ1Bit().build().alwaysUseMemoryOptimizedSearch());
        assertTrue(
            "SQ 1-bit Lucene should always use memory optimized search (matches engine-agnostic behavior on main)",
            baseFaissSQ1Bit().engine(KNNEngine.LUCENE).build().alwaysUseMemoryOptimizedSearch()
        );
        assertFalse(
            "SQ 1-bit Faiss IVF should not always use memory optimized search (IVF layout cannot be loaded by the MOS reader)",
            baseFaissSQ1Bit().methodName(METHOD_IVF).build().alwaysUseMemoryOptimizedSearch()
        );
        assertFalse(
            "SQ 16-bit should not always use memory optimized search",
            baseFaiss().encoderType(Encoder.EncoderType.SQ)
                .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
                .compressionLevel(CompressionLevel.x2)
                .build()
                .alwaysUseMemoryOptimizedSearch()
        );
        assertFalse("Flat encoder should not always use memory optimized search", baseFaiss().build().alwaysUseMemoryOptimizedSearch());
    }

    // --- Coverage: isFaissSQOneBit is Faiss-only ---

    public void testIsFaissSQOneBit_FaissSQ1Bit_true() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.isFaissSQOneBit());
    }

    public void testIsFaissSQOneBit_LuceneSQ1Bit_false() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().engine(KNNEngine.LUCENE).build();
        assertFalse(spec.isFaissSQOneBit());
    }

    // --- Coverage: isSQOneBit public method ---

    public void testIsSQOneBit_true() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.isSQOneBit());
    }

    public void testIsSQOneBit_false_forSQ16() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.x2)
            .build();
        assertFalse(spec.isSQOneBit());
    }

    public void testIsSQOneBit_false_forFlat() {
        ResolvedIndexSpec spec = baseFaiss().build();
        assertFalse(spec.isSQOneBit());
    }

    // --- Remote index build support ---

    public void testSupportsRemoteIndexBuild_NonFaissEngineNotSupported() {
        ResolvedIndexSpec spec = baseFaiss().engine(KNNEngine.LUCENE).encoderType(Encoder.EncoderType.FLAT).build();
        assertFalse(spec.supportsRemoteIndexBuild());
    }

    public void testSupportsRemoteIndexBuild_FaissHNSWFlatSupported() {
        ResolvedIndexSpec spec = baseFaiss().encoderType(Encoder.EncoderType.FLAT).build();
        assertTrue(spec.supportsRemoteIndexBuild());
    }

    private ResolvedIndexSpec.ResolvedIndexSpecBuilder baseFaiss() {
        return ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .mode(Mode.ON_DISK)
            .indexVersionCreated(Version.CURRENT);
    }

    private ResolvedIndexSpec.ResolvedIndexSpecBuilder baseFaissSQ1Bit() {
        return baseFaiss().encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32);
    }
}
