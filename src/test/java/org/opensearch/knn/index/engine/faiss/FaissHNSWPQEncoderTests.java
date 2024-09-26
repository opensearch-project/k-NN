/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class FaissHNSWPQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        FaissHNSWPQEncoder encoder = new FaissHNSWPQEncoder();
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(null, null));
    }
}
