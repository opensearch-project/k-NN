/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_INT8;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;

public class FaissSQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(generateMethodComponentContext(FAISS_SQ_ENCODER_FP16), null));
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(generateMethodComponentContext(FAISS_SQ_ENCODER_INT8), null));
    }

    private MethodComponentContext generateMethodComponentContext(String sqType) {
        return new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, sqType));
    }
}
