/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class LuceneSQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(CompressionLevel.x4, encoder.calculateCompressionLevel(null, null));
    }
}
