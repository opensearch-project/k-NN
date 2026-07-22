/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class FaissSQEncoderTests extends KNNTestCase {

    // --- Legacy (no bits) ---

    public void testCalculateCompressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(null, null));
    }

    public void testNoBits_compressionLevel_legacy() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, "fp16"));
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(mcc, null));
    }

    // --- bits=1 (1-bit quantization) ---

    public void testBits1_libraryIndexingContext() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(mcc, context);

        Map<String, Object> params = indexingContext.getLibraryParameters();
        assertEquals(FAISS_FLAT_DESCRIPTION, params.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(ENCODER_SQ, params.get("name"));
        assertEquals(1, params.get(SQ_BITS));
    }

    public void testBits1_compressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testBits16_compressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(mcc, null));
    }

    // --- Validation: bits required on 3.6.0+ ---

    public void testValidate_whenV360NoBits_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, Map.of(FAISS_SQ_TYPE, "fp16"))
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("bits"));
        assertTrue(output.getErrorMessage().contains("required"));
    }

    public void testValidate_whenPreV360NoBits_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED, Map.of(FAISS_SQ_TYPE, "fp16"))
        );
        assertNull(output.getValid());
    }

    // --- Validation: bits=1 + type not allowed ---

    public void testValidate_whenBits1WithType_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, Map.of(SQ_BITS, 1, FAISS_SQ_TYPE, "fp16"))
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("type"));
        assertTrue(output.getErrorMessage().contains("not supported"));
    }

    // --- Validation: bits=1 + clip not allowed ---

    public void testValidate_whenBits1WithClip_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, Map.of(SQ_BITS, 1, FAISS_SQ_CLIP, true))
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("clip"));
    }

    public void testValidate_whenBits16WithClip_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(SQ_BITS, 16, FAISS_SQ_TYPE, "fp16", FAISS_SQ_CLIP, true)
            )
        );
        assertNull(output.getValid());
    }

    // --- Validation: compression level compatibility ---

    public void testValidate_whenBits1WithX32Compression_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x32, Map.of(SQ_BITS, 1))
        );
        assertNull(output.getValid());
    }

    public void testValidate_whenBits1WithX2Compression_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x2, Map.of(SQ_BITS, 1))
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("incompatible"));
        assertTrue(output.getErrorMessage().contains("32x"));
    }

    public void testValidate_whenBits16WithX2Compression_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x2, Map.of(SQ_BITS, 16))
        );
        assertNull(output.getValid());
    }

    // --- Validation: invalid bits value ---

    public void testValidate_whenInvalidBits_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        // Valid bits are {1, 2, 4, 16}. Use 3 as an unambiguously invalid value.
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 3));
        assertNotNull(methodComponent.validate(mcc, context));
    }

    // --- Helper ---

    private TrainingConfigValidationInput buildValidationInput(
        Version version,
        CompressionLevel compressionLevel,
        Map<String, Object> encoderParams
    ) {
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(version)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(compressionLevel)
            .build();

        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, new HashMap<>(encoderParams));
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );

        return TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build();
    }

    public void testBits1_quantizationConfigIsEmpty() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(mcc, context);

        assertEquals(org.opensearch.knn.index.engine.qframe.QuantizationConfig.EMPTY, indexingContext.getQuantizationConfig());
    }

    public void testBits16_libraryIndexingContextUsesSQDescription() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(mcc, context);

        Map<String, Object> params = indexingContext.getLibraryParameters();
        assertNotEquals(FAISS_FLAT_DESCRIPTION, params.get(INDEX_DESCRIPTION_PARAMETER));
    }

    // --- isSQOneBit utility ---

    public void testIsSQOneBit_whenSQWithBits1_thenTrue() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertTrue(FaissSQEncoder.isSQOneBit(params));
    }

    public void testIsSQOneBit_whenSQWithBits16_thenFalse() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertFalse(FaissSQEncoder.isSQOneBit(params));
    }

    public void testIsSQOneBit_whenNonSQEncoder_thenFalse() {
        MethodComponentContext encoderCtx = new MethodComponentContext("flat", Map.of());
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertFalse(FaissSQEncoder.isSQOneBit(params));
    }

    public void testIsSQOneBit_whenNullParams_thenFalse() {
        assertFalse(FaissSQEncoder.isSQOneBit(null));
    }

    public void testIsSQOneBit_whenNoEncoder_thenFalse() {
        assertFalse(FaissSQEncoder.isSQOneBit(Map.of()));
    }

    public void testIsSQOneBit_whenEncoderNotMethodComponentContext_thenFalse() {
        assertFalse(FaissSQEncoder.isSQOneBit(Map.of(METHOD_ENCODER_PARAMETER, "not_a_context")));
    }

    // --- bits=2 / bits=4 (multi-bit MOS quantization) ---

    public void testBits2_libraryIndexingContext() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 2));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(mcc, context);

        Map<String, Object> params = indexingContext.getLibraryParameters();
        assertEquals(FAISS_FLAT_DESCRIPTION, params.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(ENCODER_SQ, params.get("name"));
        assertEquals(2, params.get(SQ_BITS));
    }

    public void testBits4_libraryIndexingContext() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 4));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(mcc, context);

        Map<String, Object> params = indexingContext.getLibraryParameters();
        assertEquals(FAISS_FLAT_DESCRIPTION, params.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(ENCODER_SQ, params.get("name"));
        assertEquals(4, params.get(SQ_BITS));
    }

    public void testBits2_compressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 2));
        assertEquals(CompressionLevel.x16, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testBits4_compressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 4));
        assertEquals(CompressionLevel.x8, encoder.calculateCompressionLevel(mcc, null));
    }

    // --- isMosBits utility ---

    public void testIsMosBits() {
        assertTrue(FaissSQEncoder.isMosBits(1));
        assertTrue(FaissSQEncoder.isMosBits(2));
        assertTrue(FaissSQEncoder.isMosBits(4));
        // fp16 is SQ but not the MOS bit-plane path
        assertFalse(FaissSQEncoder.isMosBits(16));
        for (int bits : new int[] { 0, 3, 5, 7, 8, -1 }) {
            assertFalse("Expected " + bits + " to not be MOS bits", FaissSQEncoder.isMosBits(bits));
        }
    }

    // --- getSQBits utility ---

    public void testGetSQBits_whenSQEncoder_thenReturnsBits() {
        for (int bits : new int[] { 1, 2, 4, 16 }) {
            MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, bits));
            Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
            assertEquals(Integer.valueOf(bits), FaissSQEncoder.getSQBits(params));
        }
    }

    public void testGetSQBits_whenNoBits_thenNull() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, "fp16"));
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertNull(FaissSQEncoder.getSQBits(params));
    }

    public void testGetSQBits_whenNonSQEncoder_thenNull() {
        MethodComponentContext encoderCtx = new MethodComponentContext("flat", Map.of());
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertNull(FaissSQEncoder.getSQBits(params));
    }

    public void testGetSQBits_whenNullParams_thenNull() {
        assertNull(FaissSQEncoder.getSQBits(null));
    }

    public void testGetSQBits_whenNoEncoder_thenNull() {
        assertNull(FaissSQEncoder.getSQBits(Map.of()));
    }

    public void testGetSQBits_whenEncoderNotMethodComponentContext_thenNull() {
        assertNull(FaissSQEncoder.getSQBits(Map.of(METHOD_ENCODER_PARAMETER, "not_a_context")));
    }

    // --- isSQMultiBit utility ---

    public void testIsSQMultiBit_whenSQWithMosBits_thenTrue() {
        for (int bits : new int[] { 1, 2, 4 }) {
            MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, bits));
            Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
            assertTrue("Expected bits=" + bits + " to be SQ multi-bit", FaissSQEncoder.isSQMultiBit(params));
        }
    }

    public void testIsSQMultiBit_whenSQWithBits16_thenFalse() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertFalse(FaissSQEncoder.isSQMultiBit(params));
    }

    public void testIsSQMultiBit_whenNonSQEncoder_thenFalse() {
        MethodComponentContext encoderCtx = new MethodComponentContext("flat", Map.of());
        Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderCtx);
        assertFalse(FaissSQEncoder.isSQMultiBit(params));
    }

    public void testIsSQMultiBit_whenNullParams_thenFalse() {
        assertFalse(FaissSQEncoder.isSQMultiBit(null));
    }

    // --- Bits enum ---

    public void testBitsEnum_twoAndFour() {
        assertEquals(2, FaissSQEncoder.Bits.TWO.getValue());
        assertEquals(CompressionLevel.x16, FaissSQEncoder.Bits.TWO.getCompressionLevel());

        assertEquals(4, FaissSQEncoder.Bits.FOUR.getValue());
        assertEquals(CompressionLevel.x8, FaissSQEncoder.Bits.FOUR.getCompressionLevel());
    }

    public void testBitsEnum_fromValue() {
        assertEquals(FaissSQEncoder.Bits.ONE, FaissSQEncoder.Bits.fromValue(1));
        assertEquals(FaissSQEncoder.Bits.TWO, FaissSQEncoder.Bits.fromValue(2));
        assertEquals(FaissSQEncoder.Bits.FOUR, FaissSQEncoder.Bits.fromValue(4));
        assertEquals(FaissSQEncoder.Bits.SIXTEEN, FaissSQEncoder.Bits.fromValue(16));
        expectThrows(IllegalArgumentException.class, () -> FaissSQEncoder.Bits.fromValue(3));
        expectThrows(IllegalArgumentException.class, () -> FaissSQEncoder.Bits.fromValue(8));
    }
}
