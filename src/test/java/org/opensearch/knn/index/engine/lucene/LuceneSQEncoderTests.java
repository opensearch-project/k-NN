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
import org.opensearch.common.ValidationException;
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
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> callValidateEncoderParams(Version.V_3_6_0, CompressionLevel.NOT_CONFIGURED, Map.of())
        );
        assertTrue(e.getMessage().contains("bits"));
        assertTrue(e.getMessage().contains("required"));
    }

    public void testValidate_whenPreV360NoBits_thenOk() {
        callValidateEncoderParams(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED, Map.of());
    }

    public void testValidate_WhenPreV360Bits1_thenError() {
        expectThrows(
            ValidationException.class,
            () -> callValidateEncoderParams(Version.V_3_5_0, CompressionLevel.NOT_CONFIGURED, Map.of(LUCENE_SQ_BITS, 1))
        );
    }

    public void testValidate_whenBits1WithX32Compression_thenOk() {
        callValidateEncoderParams(Version.CURRENT, CompressionLevel.x32, Map.of(LUCENE_SQ_BITS, 1));
    }

    public void testValidate_whenBits1WithConfidenceInterval_thenError() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> callValidateEncoderParams(
                Version.CURRENT,
                CompressionLevel.NOT_CONFIGURED,
                Map.of(LUCENE_SQ_BITS, 1, LUCENE_SQ_CONFIDENCE_INTERVAL, 1.0f)
            )
        );
        assertTrue(e.getMessage().contains("confidence_interval"));
        assertTrue(e.getMessage().contains("does not use additional parameter"));
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
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> callValidateEncoderParams(Version.CURRENT, CompressionLevel.x4, Map.of(LUCENE_SQ_BITS, 1))
        );
        assertTrue(e.getMessage().contains("incompatible"));
        assertTrue(e.getMessage().contains("32x"));
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
        callValidateEncoderParams(Version.CURRENT, CompressionLevel.x4, Map.of(LUCENE_SQ_BITS, 7));
    }

    public void testCalculateCompressionLevel_whenBits1InMethodComponentContext_thenX32() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 1));
        KNNMethodConfigContext context = KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build();
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(mcc, context));
    }

    public void testCalculateCompressionLevel_whenBits7InMethodComponentContext_thenX4() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 7));
        KNNMethodConfigContext context = KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build();
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(mcc, context));
    }

    public void testCalculateCompressionLevel_whenBits1AndExplicitCompressionX32_thenX32() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 1));
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .compressionLevel(CompressionLevel.x32)
            .build();
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(mcc, context));
    }

    public void testValidate_whenBits1WithX32Compression_explicitOnDisk_thenOk() {
        callValidateEncoderParams(Version.CURRENT, CompressionLevel.x32, Map.of(LUCENE_SQ_BITS, 1));
    }

    private void callValidateEncoderParams(Version version, CompressionLevel compressionLevel, Map<String, Object> encoderParams) {
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

        LuceneHNSWMethodResolver.validateEncoderParams(methodContext, configContext);
    }
}
