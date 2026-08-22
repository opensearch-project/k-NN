/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;

public class EncoderTypeTests extends KNNTestCase {

    public void testFromNameRoundTrips() {
        for (Encoder.EncoderType type : Encoder.EncoderType.values()) {
            assertEquals(type, Encoder.EncoderType.fromName(type.getName()));
        }
    }

    public void testGetNameMatchesKNNConstants() {
        assertEquals(ENCODER_FLAT, Encoder.EncoderType.FLAT.getName());
        assertEquals(ENCODER_SQ, Encoder.EncoderType.SQ.getName());
        assertEquals(ENCODER_PQ, Encoder.EncoderType.PQ.getName());
        assertEquals(ENCODER_BINARY, Encoder.EncoderType.BQ.getName());
    }

    public void testFromNameThrowsForUnsupported() {
        expectThrows(IllegalArgumentException.class, () -> Encoder.EncoderType.fromName("nonexistent"));
    }
}
