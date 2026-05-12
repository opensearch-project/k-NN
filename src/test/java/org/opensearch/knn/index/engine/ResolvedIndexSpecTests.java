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

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;

public class ResolvedIndexSpecTests extends KNNTestCase {

    // --- SQ 1-bit codec / memopt ---

    public void testFaissSQ1BitUsesSQ1BitCodecFormat() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.usesSQ1BitCodecFormat());
        assertTrue(spec.alwaysUseMemoryOptimizedSearch());
        assertTrue(spec.isMemoryOptimizedEligible());
        assertTrue(spec.requiresRescore());
    }

    public void testLuceneSQ1BitDoesNotUseSQ1BitCodecFormat() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().engine(KNNEngine.LUCENE).build();
        assertFalse(spec.usesSQ1BitCodecFormat());
        assertFalse(spec.alwaysUseMemoryOptimizedSearch());
    }

    // --- Memory optimized eligibility ---

    public void testFaissHNSWSQIsMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.isMemoryOptimizedEligible());
    }

    public void testFaissHNSWBQIsMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.TWO)
            .compressionLevel(CompressionLevel.x16)
            .build();
        assertTrue(spec.isMemoryOptimizedEligible());
    }

    public void testFaissIVFNotMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss()
            .methodName(METHOD_IVF)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertFalse(spec.isMemoryOptimizedEligible());
    }

    public void testFaissPQNotMemoryOptimizedEligible() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.PQ)
            .quantizationBits(Encoder.QuantizationBits.NOT_APPLICABLE)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.isMemoryOptimizedEligible());
    }

    // --- Radial search support ---

    public void testRadialSearch_NMSLIBNotSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .engine(KNNEngine.NMSLIB)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_BinaryDataTypeNotSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .vectorDataType(VectorDataType.BINARY)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_BQNotSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.BQ)
            .quantizationBits(Encoder.QuantizationBits.TWO)
            .compressionLevel(CompressionLevel.x16)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_QuantizedSQ4BitNotSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_QuantizedPQNotSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.PQ)
            .quantizationBits(Encoder.QuantizationBits.NOT_APPLICABLE)
            .compressionLevel(CompressionLevel.x8)
            .build();
        assertFalse(spec.supportsRadialSearch());
    }

    public void testRadialSearch_SQ1BitSupported() {
        ResolvedIndexSpec spec = baseFaissSQ1Bit().build();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_FlatMethodWithX32Supported() {
        ResolvedIndexSpec spec = baseFaiss()
            .methodName(METHOD_FLAT)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x32)
            .build();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_NonQuantizedAlwaysSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x1)
            .build();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_LuceneNonQuantizedSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .engine(KNNEngine.LUCENE)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SEVEN)
            .compressionLevel(CompressionLevel.x4)
            .build();
        assertTrue(spec.supportsRadialSearch());
    }

    public void testRadialSearch_NotConfiguredCompressionSupported() {
        ResolvedIndexSpec spec = baseFaiss()
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.SIXTEEN)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        assertTrue(spec.supportsRadialSearch());
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

    // --- Helpers ---

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
        return baseFaiss()
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32);
    }
}
