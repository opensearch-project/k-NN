/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
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
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_BF16;
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

    // --- Validation: clip not allowed for bf16 (it is a no-op since bf16 shares float32's range) ---

    public void testValidate_whenBf16WithClipTrue_thenError() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(SQ_BITS, 16, FAISS_SQ_TYPE, FAISS_SQ_ENCODER_BF16, FAISS_SQ_CLIP, true)
            )
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("clip"));
        assertTrue(output.getErrorMessage().contains("fp16"));
    }

    public void testValidate_whenBf16WithClipFalse_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(SQ_BITS, 16, FAISS_SQ_TYPE, FAISS_SQ_ENCODER_BF16, FAISS_SQ_CLIP, false)
            )
        );
        assertNull(output.getValid());
    }

    public void testValidate_whenBf16WithoutClip_thenOk() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(SQ_BITS, 16, FAISS_SQ_TYPE, FAISS_SQ_ENCODER_BF16)
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

    // --- validate() direct tests ---

    public void testValidateDirectly_whenInvalidBits_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 99)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED)
            )
        );
        assertTrue(e.getMessage().contains("Unsupported bits value"));
    }

    public void testValidateDirectly_whenV360NoBitsFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(FAISS_SQ_TYPE, "fp16")),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED)
            )
        );
        assertTrue(e.getMessage().contains("required"));
    }

    public void testValidateDirectly_whenV360WithTypeOnBits1_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 1, FAISS_SQ_TYPE, "fp16")),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED)
            )
        );
        assertTrue(e.getMessage().contains("type"));
    }

    public void testValidateDirectly_whenV360WithClipOnBits1_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 1, FAISS_SQ_CLIP, true)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED)
            )
        );
        assertTrue(e.getMessage().contains("clip"));
    }

    public void testValidateDirectly_whenValidConfig_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(buildMethodContext(Map.of(SQ_BITS, 1)), buildConfigContext(Version.CURRENT, CompressionLevel.x32));
    }

    public void testValidateDirectly_whenNullInputs_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(null, null);
    }

    public void testValidateDirectly_whenPreV360NoBits_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(
            buildMethodContext(Map.of(FAISS_SQ_TYPE, "fp16")),
            buildConfigContext(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED)
        );
    }

    public void testValidateDirectly_whenNullConfigContext_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(buildMethodContext(Map.of(SQ_BITS, 1)), null);
    }

    public void testValidateDirectly_whenEncoderContextMissing_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );
        encoder.validate(methodContext, buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED));
    }

    public void testValidateDirectly_whenNullVersionNoBits_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(buildMethodContext(Map.of(FAISS_SQ_TYPE, "fp16")), buildConfigContext(null, CompressionLevel.NOT_CONFIGURED));
    }

    public void testValidateDirectly_whenV360NoBitsByteDataType_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BYTE)
            .dimension(128)
            .compressionLevel(CompressionLevel.NOT_CONFIGURED)
            .build();
        encoder.validate(buildMethodContext(Map.of(FAISS_SQ_TYPE, "fp16")), configContext);
    }

    public void testValidateDirectly_whenBits16WithNoCompressionConfigured_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(
            buildMethodContext(Map.of(SQ_BITS, 16, FAISS_SQ_TYPE, "fp16", FAISS_SQ_CLIP, true)),
            buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED)
        );
    }

    // bits=1 on half_float takes 16 bits down to 1, so it pairs with x16. x32 is what the same encoder
    // achieves on FLOAT's 32 bits and is not a valid pairing here.
    public void testValidateDirectly_whenBits1WithHalfFloat_thenNoException() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        encoder.validate(
            buildMethodContext(Map.of(SQ_BITS, 1)),
            buildConfigContext(Version.CURRENT, CompressionLevel.x16, VectorDataType.HALF_FLOAT)
        );
    }

    public void testValidateDirectly_whenBits1WithHalfFloatAndX32_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 1)),
                buildConfigContext(Version.CURRENT, CompressionLevel.x32, VectorDataType.HALF_FLOAT)
            )
        );
        assertTrue(e.getMessage().contains("16x"));
    }

    public void testCalculateCompressionLevel_whenBits1_thenX16ForHalfFloatAndX32ForFloat() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(
            CompressionLevel.x16,
            encoder.calculateCompressionLevel(
                new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.HALF_FLOAT)
            )
        );
        assertEquals(
            CompressionLevel.x32,
            encoder.calculateCompressionLevel(
                new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.FLOAT)
            )
        );
    }

    public void testValidateDirectly_whenBits16WithHalfFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 16)),
                buildConfigContext(Version.CURRENT, CompressionLevel.x2, VectorDataType.HALF_FLOAT)
            )
        );
        assertTrue(e.getMessage().contains("half_float"));
    }

    public void testValidateDirectly_whenNoBitsWithHalfFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of()),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.HALF_FLOAT)
            )
        );
        assertTrue(e.getMessage().contains("half_float"));
    }

    public void testValidateDirectly_whenBits2WithHalfFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 2)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.HALF_FLOAT)
            )
        );
        assertTrue(e.getMessage().contains("half_float"));
    }

    public void testValidateDirectly_whenBits4WithHalfFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> encoder.validate(
                buildMethodContext(Map.of(SQ_BITS, 4)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.HALF_FLOAT)
            )
        );
        assertTrue(e.getMessage().contains("half_float"));
    }

    public void testCalculateCompressionLevel_whenBits2WithHalfFloat_thenThrows() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        expectThrows(
            IllegalArgumentException.class,
            () -> encoder.calculateCompressionLevel(
                new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 2)),
                buildConfigContext(Version.CURRENT, CompressionLevel.NOT_CONFIGURED, VectorDataType.HALF_FLOAT)
            )
        );
    }

    private KNNMethodContext buildMethodContext(Map<String, Object> encoderParams) {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, new HashMap<>(encoderParams));
        return new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );
    }

    private KNNMethodConfigContext buildConfigContext(Version version, CompressionLevel compressionLevel) {
        return buildConfigContext(version, compressionLevel, VectorDataType.FLOAT);
    }

    private KNNMethodConfigContext buildConfigContext(Version version, CompressionLevel compressionLevel, VectorDataType vectorDataType) {
        return KNNMethodConfigContext.builder()
            .versionCreated(version)
            .vectorDataType(vectorDataType)
            .dimension(128)
            .compressionLevel(compressionLevel)
            .build();
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

    // --- isSQCodedBits utility ---

    public void testIsSQCodedBits() {
        assertTrue(FaissSQEncoder.isSQCodedBits(1));
        assertTrue(FaissSQEncoder.isSQCodedBits(2));
        assertTrue(FaissSQEncoder.isSQCodedBits(4));
        // fp16 is SQ but stores compressed floats, not integer-coded bits
        assertFalse(FaissSQEncoder.isSQCodedBits(16));
        for (int bits : new int[] { 0, 3, 5, 7, 8, -1 }) {
            assertFalse("Expected " + bits + " to not be SQ-coded bits", FaissSQEncoder.isSQCodedBits(bits));
        }
    }

}
