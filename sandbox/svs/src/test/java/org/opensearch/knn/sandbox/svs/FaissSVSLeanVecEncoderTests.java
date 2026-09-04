/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LEANVEC;
import static org.opensearch.knn.sandbox.svs.SVSConstants.LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD;
import static org.opensearch.knn.sandbox.svs.SVSConstants.LEANVEC_DEFAULT_TRAINING_THRESHOLD;

public class FaissSVSLeanVecEncoderTests extends OpenSearchTestCase {

    public void testValidateBitCombination_acceptsSupported() {
        FaissSVSLeanVecEncoder.validateBitCombination(4, 4);
        FaissSVSLeanVecEncoder.validateBitCombination(4, 8);
        FaissSVSLeanVecEncoder.validateBitCombination(8, 8);
    }

    public void testValidateBitCombination_rejectsUnsupported() {
        IllegalArgumentException e40 = expectThrows(
            IllegalArgumentException.class,
            () -> FaissSVSLeanVecEncoder.validateBitCombination(4, 0)
        );
        assertTrue(e40.getMessage().contains("4x0"));
        assertTrue(e40.getMessage().contains(FAISS_SVS_ENCODER_LEANVEC));

        expectThrows(IllegalArgumentException.class, () -> FaissSVSLeanVecEncoder.validateBitCombination(2, 2));
        expectThrows(IllegalArgumentException.class, () -> FaissSVSLeanVecEncoder.validateBitCombination(8, 0));
        expectThrows(IllegalArgumentException.class, () -> FaissSVSLeanVecEncoder.validateBitCombination(8, 4));
    }

    public void testValidateThresholds_acceptsOrderedAndDefaults() {
        // Explicit ordered pair, equal pair, and 0 (= use default) in either position.
        FaissSVSLeanVecEncoder.validateThresholds(1000, 5000);
        FaissSVSLeanVecEncoder.validateThresholds(5000, 5000);
        FaissSVSLeanVecEncoder.validateThresholds(0, 0);
        FaissSVSLeanVecEncoder.validateThresholds(0, LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD);
        FaissSVSLeanVecEncoder.validateThresholds(LEANVEC_DEFAULT_TRAINING_THRESHOLD, 0);
    }

    public void testValidateThresholds_rejectsRoughAboveFinal() {
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FaissSVSLeanVecEncoder.validateThresholds(5000, 1000)
        );
        assertTrue(e.getMessage().contains("rough_training_threshold"));

        // rough default (10K) above an explicit final of 1000 must also be rejected.
        expectThrows(IllegalArgumentException.class, () -> FaissSVSLeanVecEncoder.validateThresholds(0, 1000));
    }

    public void testCompressionLevel_notConfigured() {
        FaissSVSLeanVecEncoder encoder = new FaissSVSLeanVecEncoder();
        assertEquals(
            CompressionLevel.NOT_CONFIGURED,
            encoder.calculateCompressionLevel(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of()), null)
        );
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(null, null));
    }

    /**
     * The reduced dimensionality must not exceed the field dimension: above it the mapping would be accepted
     * and only fail inside native training at the first merge over the training threshold.
     */
    public void testDimensions_boundedByFieldDimension() {
        MethodComponent component = new FaissSVSLeanVecEncoder().getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .versionCreated(Version.CURRENT)
            .build();
        assertNotNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", 1024)), context));
        assertNotNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", 129)), context));
        assertNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", 128)), context));
        assertNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", 64)), context));
        assertNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", 0)), context));
        assertNotNull(component.validate(new MethodComponentContext(FAISS_SVS_ENCODER_LEANVEC, Map.of("dimensions", -1)), context));
    }

}
