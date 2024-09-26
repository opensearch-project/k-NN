/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class FaissSQEncoderTests extends KNNTestCase {
    public void testCalculateCompressionLevel() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(CompressionLevel.x2, encoder.calculateCompressionLevel(null, null));
    }
}
