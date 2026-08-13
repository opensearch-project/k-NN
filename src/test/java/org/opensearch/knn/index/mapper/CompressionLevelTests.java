/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.core.common.Strings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.Encoder.QuantizationBits;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.knn.index.engine.lucene.LuceneSQEncoder;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

public class CompressionLevelTests extends KNNTestCase {

    public void testFromName() {
        assertEquals(CompressionLevel.NOT_CONFIGURED, CompressionLevel.fromName(null));
        assertEquals(CompressionLevel.NOT_CONFIGURED, CompressionLevel.fromName(""));
        assertEquals(CompressionLevel.x1, CompressionLevel.fromName("1x"));
        assertEquals(CompressionLevel.x32, CompressionLevel.fromName("32x"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("x1"));
    }

    public void testGetName() {
        assertTrue(Strings.isEmpty(CompressionLevel.NOT_CONFIGURED.getName()));
        assertEquals("4x", CompressionLevel.x4.getName());
        assertEquals("16x", CompressionLevel.x16.getName());
    }

    public void testNumBitsForFloat32() {
        assertEquals(1, CompressionLevel.x32.numBitsForFloat32());
        assertEquals(2, CompressionLevel.x16.numBitsForFloat32());
        assertEquals(4, CompressionLevel.x8.numBitsForFloat32());
        assertEquals(8, CompressionLevel.x4.numBitsForFloat32());
        assertEquals(16, CompressionLevel.x2.numBitsForFloat32());
        assertEquals(32, CompressionLevel.x1.numBitsForFloat32());
        assertEquals(32, CompressionLevel.NOT_CONFIGURED.numBitsForFloat32());
    }

    public void testIsConfigured() {
        assertFalse(CompressionLevel.isConfigured(CompressionLevel.NOT_CONFIGURED));
        assertFalse(CompressionLevel.isConfigured(null));
        assertTrue(CompressionLevel.isConfigured(CompressionLevel.x1));
    }

    public void testIsModeValidForRescore() {
        assertTrue(CompressionLevel.x32.isModeValidForRescore(Mode.ON_DISK));
        assertTrue(CompressionLevel.x16.isModeValidForRescore(Mode.ON_DISK));
        assertTrue(CompressionLevel.x8.isModeValidForRescore(Mode.ON_DISK));
        assertTrue(CompressionLevel.x4.isModeValidForRescore(Mode.ON_DISK));
        assertFalse(CompressionLevel.x2.isModeValidForRescore(Mode.ON_DISK));
        assertFalse(CompressionLevel.x1.isModeValidForRescore(Mode.ON_DISK));
        assertFalse(CompressionLevel.NOT_CONFIGURED.isModeValidForRescore(Mode.ON_DISK));
        assertFalse(CompressionLevel.x32.isModeValidForRescore(Mode.IN_MEMORY));
    }

    public void testGetDefaultRescoreContextForLevel() {
        assertNotNull(CompressionLevel.x32.getDefaultRescoreContext());
        assertEquals(3.0f, CompressionLevel.x32.getDefaultRescoreContext().getOversampleFactor(), 0.0f);
        assertEquals(3.0f, CompressionLevel.x16.getDefaultRescoreContext().getOversampleFactor(), 0.0f);
        assertEquals(2.0f, CompressionLevel.x8.getDefaultRescoreContext().getOversampleFactor(), 0.0f);
        assertEquals(1.0f, CompressionLevel.x4.getDefaultRescoreContext().getOversampleFactor(), 0.0f);
        assertNull(CompressionLevel.x2.getDefaultRescoreContext());
        assertNull(CompressionLevel.x1.getDefaultRescoreContext());
        assertNull(CompressionLevel.NOT_CONFIGURED.getDefaultRescoreContext());
    }

    public void testGetRescoreContext_viaResolvedIndexSpec() {
        int belowThresholdDimension = 500;
        int aboveThresholdDimension = 1500;

        // x32 with dimension <= 1000 should have an oversample factor of 5.0f
        RescoreContext rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x32 with dimension > 1000 should have an oversample factor of 3.0f
        rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x16 with dimension <= 1000 should have an oversample factor of 5.0f
        rescoreContext = buildSpec(CompressionLevel.x16, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x16 with dimension > 1000 should have an oversample factor of 3.0f
        rescoreContext = buildSpec(CompressionLevel.x16, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(3.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x8 with dimension <= 1000 should have an oversample factor of 5.0f
        rescoreContext = buildSpec(CompressionLevel.x8, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x8 with dimension > 1000 should have an oversample factor of 2.0f
        rescoreContext = buildSpec(CompressionLevel.x8, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x4 with dimension <= 1000 should have an oversample factor of 1.0f
        rescoreContext = buildSpec(CompressionLevel.x4, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x4 with dimension > 1000 should have an oversample factor of 1.0f
        rescoreContext = buildSpec(CompressionLevel.x4, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(1.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertTrue(rescoreContext.isRescoreEnabled());
        assertFalse(rescoreContext.isUserProvided());

        // x2 should return null (no rescore for x2)
        rescoreContext = buildSpec(CompressionLevel.x2, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNull(rescoreContext);
        rescoreContext = buildSpec(CompressionLevel.x2, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNull(rescoreContext);

        // x1 should return null
        rescoreContext = buildSpec(CompressionLevel.x1, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNull(rescoreContext);
        rescoreContext = buildSpec(CompressionLevel.x1, Mode.ON_DISK, aboveThresholdDimension, null).getRescoreContext();
        assertNull(rescoreContext);

        // NOT_CONFIGURED should return null
        rescoreContext = buildSpec(CompressionLevel.NOT_CONFIGURED, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNull(rescoreContext);

        // x32 with Lucene engine and dimension <= 1000 should use 2.0f (Lucene scalar quantizer)
        rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, belowThresholdDimension, KNNEngine.LUCENE).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());

        // x32 with Lucene engine and dimension > 1000
        rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, aboveThresholdDimension, KNNEngine.LUCENE).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());

        // x32 with Faiss engine should return default behavior (not special Lucene handling)
        rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, belowThresholdDimension, KNNEngine.FAISS).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());

        // x32 with null engine should return default behavior
        rescoreContext = buildSpec(CompressionLevel.x32, Mode.ON_DISK, belowThresholdDimension, null).getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(5.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());
    }

    public void testGetRescoreContext_whenFlatMethod_thenReturnFlatOversampleFactor() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_FLAT)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(2.0f, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());

        // non-flat method on x32 with NOT_CONFIGURED mode should return null (no mode for rescore)
        spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        rescoreContext = spec.getRescoreContext();
        assertNull(rescoreContext);
    }

    public void testGetRescoreContext_whenSQOneBitEncoder_thenReturnFixedOversampleFactor() {
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.NOT_CONFIGURED)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isUserProvided());
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());

        // sq(bits=1) should also work with ON_DISK mode and high dimension
        spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.ONE)
            .compressionLevel(CompressionLevel.x32)
            .mode(Mode.ON_DISK)
            .dimension(1500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR, rescoreContext.getOversampleFactor(), 0.0f);
        assertFalse(rescoreContext.isAllowOverrideOversampleFactor());
    }

    public void testGetRescoreContext_whenNonSQOneBitEncoder_thenFallsBackToNormalLogic() {
        // Non-sq(bits=1) encoder should fall through to normal compression level logic
        ResolvedIndexSpec spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.SQ)
            .quantizationBits(Encoder.QuantizationBits.FOUR)
            .compressionLevel(CompressionLevel.x8)
            .mode(Mode.ON_DISK)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        RescoreContext rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD, rescoreContext.getOversampleFactor(), 0.0f);

        // FLAT encoder should also fall through
        spec = ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(CompressionLevel.x8)
            .mode(Mode.ON_DISK)
            .dimension(500)
            .indexVersionCreated(Version.CURRENT)
            .build();
        rescoreContext = spec.getRescoreContext();
        assertNotNull(rescoreContext);
        assertEquals(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD, rescoreContext.getOversampleFactor(), 0.0f);
    }

    public void testX32MapsToOneBitQuantization() {
        assertEquals(1, CompressionLevel.x32.numBitsForFloat32());

        assertEquals(CompressionLevel.x32, QuantizationBits.ONE.getCompressionLevel());
        assertEquals(1, QuantizationBits.ONE.getValue());

        assertEquals(CompressionLevel.x32, LuceneSQEncoder.Bits.ONE.getCompressionLevel());
        assertEquals(1, LuceneSQEncoder.Bits.ONE.getValue());

        assertEquals(1, ScalarQuantizationType.ONE_BIT.getId());
    }

    public void testCompressionLevelEnumMappings() {
        assertEquals(1, CompressionLevel.x32.numBitsForFloat32());
        assertEquals("32x", CompressionLevel.x32.getName());

        assertEquals(2, CompressionLevel.x16.numBitsForFloat32());
        assertEquals("16x", CompressionLevel.x16.getName());

        assertEquals(4, CompressionLevel.x8.numBitsForFloat32());
        assertEquals("8x", CompressionLevel.x8.getName());

        assertEquals(8, CompressionLevel.x4.numBitsForFloat32());
        assertEquals("4x", CompressionLevel.x4.getName());

        assertEquals(16, CompressionLevel.x2.numBitsForFloat32());
        assertEquals("2x", CompressionLevel.x2.getName());

        assertEquals(32, CompressionLevel.x1.numBitsForFloat32());
        assertEquals("1x", CompressionLevel.x1.getName());
    }

    public void testFromNameAllValidLevels() {
        assertEquals(CompressionLevel.x1, CompressionLevel.fromName("1x"));
        assertEquals(CompressionLevel.x2, CompressionLevel.fromName("2x"));
        assertEquals(CompressionLevel.x4, CompressionLevel.fromName("4x"));
        assertEquals(CompressionLevel.x8, CompressionLevel.fromName("8x"));
        assertEquals(CompressionLevel.x16, CompressionLevel.fromName("16x"));
        assertEquals(CompressionLevel.x32, CompressionLevel.fromName("32x"));
        assertEquals(CompressionLevel.x64, CompressionLevel.fromName("64x"));
    }

    public void testDefaultCompressionIsX1() {
        // TODO: [DEFAULT_FLIP] After Step 4, assert CompressionLevel.x32 (SQ 1-bit) instead
        assertEquals(32, CompressionLevel.NOT_CONFIGURED.numBitsForFloat32());
        assertEquals(CompressionLevel.x1.numBitsForFloat32(), CompressionLevel.NOT_CONFIGURED.numBitsForFloat32());
    }

    public void testMaxCompressionLevelConstant() {
        assertEquals(CompressionLevel.x64, CompressionLevel.MAX_COMPRESSION_LEVEL);
    }

    public void testFromName_whenInvalid_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("99x"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("0x"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("x32"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("3x"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("abc"));
        expectThrows(IllegalArgumentException.class, () -> CompressionLevel.fromName("-1x"));
    }

    public void testX32ConsistentAcrossEngineEncoders() {
        QuantizationBits faissBits1 = QuantizationBits.fromValue(1);
        LuceneSQEncoder.Bits luceneBits1 = LuceneSQEncoder.Bits.fromValue(1);

        assertEquals(faissBits1.getCompressionLevel(), luceneBits1.getCompressionLevel());
        assertEquals(CompressionLevel.x32, faissBits1.getCompressionLevel());
        assertEquals(CompressionLevel.x32, luceneBits1.getCompressionLevel());
    }

    private ResolvedIndexSpec buildSpec(CompressionLevel compression, Mode mode, int dimension, KNNEngine engine) {
        return ResolvedIndexSpec.builder()
            .engine(engine != null ? engine : KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .quantizationBits(Encoder.QuantizationBits.FULL_PRECISION)
            .compressionLevel(compression)
            .mode(mode)
            .dimension(dimension)
            .indexVersionCreated(Version.CURRENT)
            .build();
    }
}
