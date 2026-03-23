/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import java.util.HashMap;
import java.util.Map;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class LuceneSQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(null, null));
    }

    public void testCalculateCompressionLevelForVersionAfter360() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(
            CompressionLevel.x32,
            encoder.calculateCompressionLevel(null, KNNMethodConfigContext.builder().versionCreated(Version.V_3_6_0).build())
        );
    }

    public void testCalculateCompressionLevelWhenUserSupplied() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();

        // Checking on a non-valid compression level to ensure users choice still taken
        assertEquals(
            CompressionLevel.x16,
            encoder.calculateCompressionLevel(
                null,
                KNNMethodConfigContext.builder().versionCreated(Version.V_3_6_0).compressionLevel(CompressionLevel.x16).build()
            )
        );
    }

    public void testValidate_WhenV360NoBits_thenError() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.V_3_6_0, CompressionLevel.NOT_CONFIGURED, Map.of())
        );

        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("bits"));
        assertTrue(output.getErrorMessage().contains("required"));
    }

    public void testValidate_whenPreV360NoBits_thenOk() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED, Map.of())
        );
        assertNull(output.getValid());
    }

    public void testValidate_WhenPreV360Bits1_thenError() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED, Map.of(LUCENE_SQ_BITS, 1))
        );

        assertNotNull(output.getValid());
        assertFalse(output.getValid());
    }

    public void testValidate_whenBits1WithX32Compression_thenOk() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x32, Map.of(LUCENE_SQ_BITS, 1))
        );
        assertNull(output.getValid());
    }

    public void testValidate_whenBits1WithConfidenceInterval_thenError() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(LUCENE_SQ_BITS, 1, LUCENE_SQ_CONFIDENCE_INTERVAL, 1.0f)
            )
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("confidence_interval"));
        assertTrue(output.getErrorMessage().contains("does not use additional parameter"));
    }

    public void testValidate_whenInvalidBits_thenError() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 2));
        assertNotNull(methodComponent.validate(mcc, context));
    }

    public void testValidate_whenBits1WithX2Compression_thenError() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x4, Map.of(LUCENE_SQ_BITS, 1))
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("incompatible"));
        assertTrue(output.getErrorMessage().contains("32x"));
    }

    public void testCalculateCompressionLevel_whenNotConfiguredPreV360() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(
            CompressionLevel.x4,
            encoder.calculateCompressionLevel(null, KNNMethodConfigContext.builder().versionCreated(Version.V_3_5_0).build())
        );
    }

    public void testCalculateCompressionLevel_whenConfiugred() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(
            CompressionLevel.x4,
            encoder.calculateCompressionLevel(
                null,
                KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).compressionLevel(CompressionLevel.x4).build()
            )
        );
    }

    public void testValidate_whenBits7WithX4Compression_thenOk() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        TrainingConfigValidationOutput output = encoder.validateEncoderConfig(
            buildValidationInput(Version.CURRENT, CompressionLevel.x4, Map.of(LUCENE_SQ_BITS, 7))
        );
        assertNull(output.getValid());
    }

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
            KNNEngine.LUCENE,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );

        return TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build();
    }
}
