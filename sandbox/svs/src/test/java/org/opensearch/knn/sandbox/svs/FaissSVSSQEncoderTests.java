/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_FP16_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_SQ8_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_FP16;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_SQ8;

public class FaissSVSSQEncoderTests extends OpenSearchTestCase {

    private KNNMethodConfigContext configContext() {
        return KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).vectorDataType(VectorDataType.FLOAT).dimension(128).build();
    }

    public void testCompressionLevel_fp16IsX2() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16));
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testCompressionLevel_sq8IsX4() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_SQ8));
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(mcc, null));
    }

    public void testCompressionLevel_defaultIsFp16() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of());
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(mcc, null));
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(null, null));
    }

    public void testInvalidType_rejected() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponent component = encoder.getMethodComponent();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, "bogus"));
        assertNotNull(component.validate(mcc, configContext()));
    }

    public void testBitsParameter_rejectedWithActionableMessage() {
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 8));
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> FaissSVSSQEncoder.validateNoBitsParameter(mcc));
        assertTrue(e.getMessage().contains("type"));
        assertTrue(e.getMessage().contains(SQ_BITS));
    }

    public void testNoBitsParameter_accepted() {
        FaissSVSSQEncoder.validateNoBitsParameter(
            new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16))
        );
        FaissSVSSQEncoder.validateNoBitsParameter(new MethodComponentContext(ENCODER_SQ, Map.of()));
        FaissSVSSQEncoder.validateNoBitsParameter(null);
    }

    public void testIndexDescription_fp16() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16));
        KNNLibraryIndexingContext ctx = encoder.getMethodComponent().getKNNLibraryIndexingContext(mcc, configContext());
        assertEquals(FAISS_SVS_SQ_FP16_DESCRIPTION, ctx.getLibraryParameters().get(INDEX_DESCRIPTION_PARAMETER));
    }

    public void testIndexDescription_sq8() {
        FaissSVSSQEncoder encoder = new FaissSVSSQEncoder();
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_SQ8));
        KNNLibraryIndexingContext ctx = encoder.getMethodComponent().getKNNLibraryIndexingContext(mcc, configContext());
        assertEquals(FAISS_SVS_SQ_SQ8_DESCRIPTION, ctx.getLibraryParameters().get(INDEX_DESCRIPTION_PARAMETER));
    }
}
