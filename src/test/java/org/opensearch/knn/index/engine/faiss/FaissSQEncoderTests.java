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

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 2));
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
}
