/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.faiss.QFrameBitEncoder.BITCOUNT_PARAM;
import static org.opensearch.knn.index.engine.faiss.QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM;

public class QFrameBitEncoderTests extends KNNTestCase {
    public void testGetLibraryIndexingContext() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        MethodComponent methodComponent = qFrameBitEncoder.getMethodComponent();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(
            QFrameBitEncoder.NAME,
            ImmutableMap.of(BITCOUNT_PARAM, 4)
        );

        KNNLibraryIndexingContext knnLibraryIndexingContext = methodComponent.getKNNLibraryIndexingContext(
            methodComponentContext,
            knnMethodConfigContext
        );
        assertEquals(
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION),
            knnLibraryIndexingContext.getLibraryParameters()
        );
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build(),
            knnLibraryIndexingContext.getQuantizationConfig()
        );

        methodComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 2));
        knnLibraryIndexingContext = methodComponent.getKNNLibraryIndexingContext(methodComponentContext, knnMethodConfigContext);
        assertEquals(
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION),
            knnLibraryIndexingContext.getLibraryParameters()
        );
        assertEquals(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build(),
            knnLibraryIndexingContext.getQuantizationConfig()
        );
    }

    public void testRandomRotationConfiguration() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        MethodComponent methodComponent = qFrameBitEncoder.getMethodComponent();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();

        // Test default random rotation (should be false)
        MethodComponentContext defaultContext = new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 4));
        KNNLibraryIndexingContext defaultConfig = methodComponent.getKNNLibraryIndexingContext(defaultContext, knnMethodConfigContext);
        assertFalse(defaultConfig.getQuantizationConfig().isEnableRandomRotation());

        // Test explicit random rotation = false
        MethodComponentContext explicitFalseContext = new MethodComponentContext(
            QFrameBitEncoder.NAME,
            ImmutableMap.of(BITCOUNT_PARAM, 4, ENABLE_RANDOM_ROTATION_PARAM, false)
        );
        KNNLibraryIndexingContext explicitFalseConfig = methodComponent.getKNNLibraryIndexingContext(
            explicitFalseContext,
            knnMethodConfigContext
        );
        assertFalse(explicitFalseConfig.getQuantizationConfig().isEnableRandomRotation());

        // Test explicit random rotation = true
        MethodComponentContext explicitTrueContext = new MethodComponentContext(
            QFrameBitEncoder.NAME,
            ImmutableMap.of(BITCOUNT_PARAM, 4, ENABLE_RANDOM_ROTATION_PARAM, true)
        );
        KNNLibraryIndexingContext explicitTrueConfig = methodComponent.getKNNLibraryIndexingContext(
            explicitTrueContext,
            knnMethodConfigContext
        );
        assertTrue(explicitTrueConfig.getQuantizationConfig().isEnableRandomRotation());
    }

    public void testInvalidRandomRotationValue() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        MethodComponent methodComponent = qFrameBitEncoder.getMethodComponent();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();

        // Test invalid random rotation value
        MethodComponentContext invalidContext = new MethodComponentContext(
            QFrameBitEncoder.NAME,
            ImmutableMap.of(BITCOUNT_PARAM, 4, ENABLE_RANDOM_ROTATION_PARAM, "invalid")
        );

        ValidationException validationException = methodComponent.validate(invalidContext, knnMethodConfigContext);

        assertTrue(
            validationException.getMessage().contains("value is not an instance of Boolean for Boolean parameter [random_rotation].")
        );
    }

    public void testRandomRotationWithDifferentBitCounts() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        MethodComponent methodComponent = qFrameBitEncoder.getMethodComponent();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();

        // Test with different bit counts
        int[] validBitCounts = { 1, 2, 4 };
        for (int bitCount : validBitCounts) {
            MethodComponentContext context = new MethodComponentContext(
                QFrameBitEncoder.NAME,
                ImmutableMap.of(BITCOUNT_PARAM, bitCount, ENABLE_RANDOM_ROTATION_PARAM, true)
            );
            KNNLibraryIndexingContext config = methodComponent.getKNNLibraryIndexingContext(context, knnMethodConfigContext);
            assertTrue(config.getQuantizationConfig().isEnableRandomRotation());

            // Verify correct bit count configuration
            ScalarQuantizationType expectedType = switch (bitCount) {
                case 1 -> ScalarQuantizationType.ONE_BIT;
                case 2 -> ScalarQuantizationType.TWO_BIT;
                case 4 -> ScalarQuantizationType.FOUR_BIT;
                default -> throw new IllegalStateException("Unexpected value: " + bitCount);
            };
            assertEquals(expectedType, config.getQuantizationConfig().getQuantizationType());
        }
    }

    public void testValidate() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        MethodComponent methodComponent = qFrameBitEncoder.getMethodComponent();

        // Invalid data type
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BYTE)
            .dimension(10)
            .build();
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            QFrameBitEncoder.NAME,
            ImmutableMap.of(BITCOUNT_PARAM, 4)
        );

        assertNotNull(methodComponent.validate(methodComponentContext, knnMethodConfigContext));

        // Invalid param
        knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();
        methodComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 4, "invalid", 4));
        assertNotNull(methodComponent.validate(methodComponentContext, knnMethodConfigContext));

        // Invalid param type
        knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();
        methodComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, "invalid"));
        assertNotNull(methodComponent.validate(methodComponentContext, knnMethodConfigContext));

        // Invalid param value
        knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(10)
            .build();
        methodComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 5));
        assertNotNull(methodComponent.validate(methodComponentContext, knnMethodConfigContext));
    }

    public void testIsTrainingRequired() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        assertFalse(
            qFrameBitEncoder.getMethodComponent()
                .isTrainingRequired(new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 4)))
        );
    }

    public void testEstimateOverheadInKB() {
        QFrameBitEncoder qFrameBitEncoder = new QFrameBitEncoder();
        assertEquals(
            0,
            qFrameBitEncoder.getMethodComponent()
                .estimateOverheadInKB(new MethodComponentContext(QFrameBitEncoder.NAME, ImmutableMap.of(BITCOUNT_PARAM, 4)), 8)
        );
    }

    public void testCalculateCompressionLevel() {
        QFrameBitEncoder encoder = new QFrameBitEncoder();
        assertEquals(
            CompressionLevel.x32,
            encoder.calculateCompressionLevel(generateMethodComponentContext(CompressionLevel.x32.numBitsForFloat32()), null)
        );
        assertEquals(
            CompressionLevel.x16,
            encoder.calculateCompressionLevel(generateMethodComponentContext(CompressionLevel.x16.numBitsForFloat32()), null)
        );
        assertEquals(
            CompressionLevel.x8,
            encoder.calculateCompressionLevel(generateMethodComponentContext(CompressionLevel.x8.numBitsForFloat32()), null)
        );
        assertEquals(
            CompressionLevel.NOT_CONFIGURED,
            encoder.calculateCompressionLevel(
                new MethodComponentContext(
                    METHOD_HNSW,
                    new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(QFrameBitEncoder.NAME, Map.of())))
                ),
                null
            )
        );

        expectThrows(
            ValidationException.class,
            () -> encoder.calculateCompressionLevel(generateMethodComponentContext(CompressionLevel.x4.numBitsForFloat32()), null)
        );
        expectThrows(ValidationException.class, () -> encoder.calculateCompressionLevel(generateMethodComponentContext(-1), null));
    }

    private MethodComponentContext generateMethodComponentContext(int bitCount) {
        return new MethodComponentContext(QFrameBitEncoder.NAME, Map.of(BITCOUNT_PARAM, bitCount));
    }
}
