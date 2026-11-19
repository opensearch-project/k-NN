/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_RABITQ_QUERY_BITS;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_RABITQ_QUERY_BITS_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_RABITQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_RABITQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class FaissIVFRaBitQEncoderTests extends KNNTestCase {

    private static final FaissIVFRaBitQEncoder ENCODER = new FaissIVFRaBitQEncoder();
    private static final MethodComponentContext ENCODER_CONTEXT = new MethodComponentContext(ENCODER_RABITQ, Collections.emptyMap());

    public void testGetName() {
        assertEquals(ENCODER_RABITQ, ENCODER.getName());
        assertEquals(Encoder.EncoderType.RABITQ, ENCODER.getEncoderType());
        assertEquals(Encoder.QuantizationBits.ONE, ENCODER.getQuantizationBits());
        assertTrue(ENCODER.getMethodComponent().isTrainingRequired(ENCODER_CONTEXT));
    }

    public void testGetLibraryIndexingContext() {
        Map<String, Object> parameters = ENCODER.getMethodComponent()
            .getKNNLibraryIndexingContext(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 128))
            .getLibraryParameters();
        assertEquals(FAISS_RABITQ_DESCRIPTION, parameters.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(Map.of(ENCODER_PARAMETER_RABITQ_QUERY_BITS, ENCODER_PARAMETER_RABITQ_QUERY_BITS_DEFAULT), parameters.get(PARAMETERS));
    }

    public void testValidate_whenQueryBits_thenOnlyZeroToEightValid() {
        MethodComponent methodComponent = ENCODER.getMethodComponent();
        KNNMethodConfigContext configContext = configContext(VectorDataType.FLOAT, 128);
        for (int queryBits : new int[] { 0, 1, 4, 8 }) {
            assertNull(methodComponent.validate(queryBitsContext(queryBits), configContext));
        }
        for (int queryBits : new int[] { -1, 9 }) {
            assertNotNull(methodComponent.validate(queryBitsContext(queryBits), configContext));
        }
    }

    public void testValidate_whenFloat_thenValid() {
        assertNull(ENCODER.getMethodComponent().validate(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 128)));
    }

    public void testValidate_whenNotFloat_thenInvalid() {
        MethodComponent methodComponent = ENCODER.getMethodComponent();
        assertNotNull(methodComponent.validate(ENCODER_CONTEXT, configContext(VectorDataType.BYTE, 128)));
        assertNotNull(methodComponent.validate(ENCODER_CONTEXT, configContext(VectorDataType.BINARY, 128)));
        assertNotNull(methodComponent.validate(ENCODER_CONTEXT, configContext(VectorDataType.HALF_FLOAT, 128)));
    }

    public void testEstimateOverheadInKB() {
        // 128 x 128 float rotation matrix = 64 KB
        assertEquals(65, ENCODER.getMethodComponent().estimateOverheadInKB(ENCODER_CONTEXT, 128));
        assertEquals(2305, ENCODER.getMethodComponent().estimateOverheadInKB(ENCODER_CONTEXT, 768));
    }

    public void testCalculateCompressionLevel() {
        // 1 bit per dimension plus 64 bits of correction factors, rounded down to a defined level
        assertEquals(CompressionLevel.x16, ENCODER.calculateCompressionLevel(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 128)));
        assertEquals(CompressionLevel.x16, ENCODER.calculateCompressionLevel(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 768)));
        assertEquals(CompressionLevel.x16, ENCODER.calculateCompressionLevel(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 16000)));
        assertEquals(CompressionLevel.x8, ENCODER.calculateCompressionLevel(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 32)));
        assertEquals(CompressionLevel.x1, ENCODER.calculateCompressionLevel(ENCODER_CONTEXT, configContext(VectorDataType.FLOAT, 2)));
        assertEquals(
            CompressionLevel.NOT_CONFIGURED,
            ENCODER.calculateCompressionLevel(
                ENCODER_CONTEXT,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build()
            )
        );
    }

    public void testValidate_whenFeatureFlagDisabled_thenThrow() {
        ValidationException e = expectThrows(
            ValidationException.class,
            () -> ENCODER.validate(null, configContext(VectorDataType.FLOAT, 128))
        );
        assertTrue(e.getMessage().contains(KNNFeatureFlags.KNN_RABITQ_ENABLED_SETTING.getKey()));
    }

    public void testValidate_whenFeatureFlagEnabled_thenNoException() {
        enableRaBitQ();
        ENCODER.validate(null, configContext(VectorDataType.FLOAT, 128));
    }

    public void testValidateEncoderConfig_whenFeatureFlagDisabled_thenInvalid() {
        TrainingConfigValidationOutput output = ENCODER.validateEncoderConfig(TrainingConfigValidationInput.builder().build());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains(KNNFeatureFlags.KNN_RABITQ_ENABLED_SETTING.getKey()));
    }

    public void testValidateEncoderConfig_whenFeatureFlagEnabled_thenNotInvalid() {
        enableRaBitQ();
        TrainingConfigValidationOutput output = ENCODER.validateEncoderConfig(TrainingConfigValidationInput.builder().build());
        assertNotEquals(Boolean.FALSE, output.getValid());
    }

    private void enableRaBitQ() {
        clusterService.getClusterSettings()
            .applySettings(Settings.builder().put(KNNFeatureFlags.KNN_RABITQ_ENABLED_SETTING.getKey(), true).build());
    }

    private static MethodComponentContext queryBitsContext(int queryBits) {
        return new MethodComponentContext(ENCODER_RABITQ, Map.of(ENCODER_PARAMETER_RABITQ_QUERY_BITS, queryBits));
    }

    private static KNNMethodConfigContext configContext(VectorDataType vectorDataType, int dimension) {
        return KNNMethodConfigContext.builder().vectorDataType(vectorDataType).dimension(dimension).versionCreated(Version.CURRENT).build();
    }
}
