/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;

public class FaissSVSLVQEncoderTests extends OpenSearchTestCase {

    public void testValidateBitCombination_acceptsSupported() {
        FaissSVSLVQEncoder.validateBitCombination(4, 0);
        FaissSVSLVQEncoder.validateBitCombination(4, 4);
        FaissSVSLVQEncoder.validateBitCombination(4, 8);
    }

    public void testValidateBitCombination_rejectsUnsupported() {
        IllegalArgumentException e88 = expectThrows(IllegalArgumentException.class, () -> FaissSVSLVQEncoder.validateBitCombination(8, 8));
        assertTrue(e88.getMessage().contains("8x8"));
        assertTrue(e88.getMessage().contains(FAISS_SVS_ENCODER_LVQ));

        expectThrows(IllegalArgumentException.class, () -> FaissSVSLVQEncoder.validateBitCombination(2, 2));
        expectThrows(IllegalArgumentException.class, () -> FaissSVSLVQEncoder.validateBitCombination(8, 0));
        expectThrows(IllegalArgumentException.class, () -> FaissSVSLVQEncoder.validateBitCombination(4, 2));
    }

    public void testCompressionLevel_4x0IsX8() {
        FaissSVSLVQEncoder encoder = new FaissSVSLVQEncoder();
        MethodComponentContext mcc = lvq(4, 0);
        assertEquals(CompressionLevel.x8, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testCompressionLevel_4x4IsX4() {
        FaissSVSLVQEncoder encoder = new FaissSVSLVQEncoder();
        MethodComponentContext mcc = lvq(4, 4);
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testCompressionLevel_4x8IsX4() {
        FaissSVSLVQEncoder encoder = new FaissSVSLVQEncoder();
        MethodComponentContext mcc = lvq(4, 8);
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testCompressionLevel_defaultIsX4() {
        FaissSVSLVQEncoder encoder = new FaissSVSLVQEncoder();
        assertEquals(
            CompressionLevel.x4,
            encoder.calculateCompressionLevel(new MethodComponentContext(FAISS_SVS_ENCODER_LVQ, Map.of()), null)
        );
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(null, null));
    }

    private static MethodComponentContext lvq(int primaryBits, int residualBits) {
        return new MethodComponentContext(
            FAISS_SVS_ENCODER_LVQ,
            Map.of(METHOD_PARAMETER_LVQ_PRIMARY_BITS, primaryBits, METHOD_PARAMETER_LVQ_RESIDUAL_BITS, residualBits)
        );
    }
}
