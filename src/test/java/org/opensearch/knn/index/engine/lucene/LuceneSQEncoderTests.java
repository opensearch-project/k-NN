/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;

public class LuceneSQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(generateMethodComponentContext(4), null));
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(generateMethodComponentContext(7), null));
    }

    private MethodComponentContext generateMethodComponentContext(int bitCount) {
        return new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, bitCount));
    }
}
