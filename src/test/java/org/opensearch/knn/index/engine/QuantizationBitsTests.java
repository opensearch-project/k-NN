/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class QuantizationBitsTests extends KNNTestCase {

    public void testFromValueRoundTrips() {
        for (Encoder.QuantizationBits bits : Encoder.QuantizationBits.values()) {
            assertEquals(bits, Encoder.QuantizationBits.fromValue(bits.getValue()));
        }
    }

    public void testGetValueMatchesExpected() {
        assertEquals(1, Encoder.QuantizationBits.ONE.getValue());
        assertEquals(2, Encoder.QuantizationBits.TWO.getValue());
        assertEquals(4, Encoder.QuantizationBits.FOUR.getValue());
        assertEquals(7, Encoder.QuantizationBits.SEVEN.getValue());
        assertEquals(16, Encoder.QuantizationBits.SIXTEEN.getValue());
        assertEquals(32, Encoder.QuantizationBits.FULL_PRECISION.getValue());
    }

    public void testGetCompressionLevelMapping() {
        assertEquals(CompressionLevel.x32, Encoder.QuantizationBits.ONE.getCompressionLevel());
        assertEquals(CompressionLevel.x16, Encoder.QuantizationBits.TWO.getCompressionLevel());
        assertEquals(CompressionLevel.x8, Encoder.QuantizationBits.FOUR.getCompressionLevel());
        assertEquals(CompressionLevel.x4, Encoder.QuantizationBits.SEVEN.getCompressionLevel());
        assertEquals(CompressionLevel.x2, Encoder.QuantizationBits.SIXTEEN.getCompressionLevel());
        assertEquals(CompressionLevel.x1, Encoder.QuantizationBits.FULL_PRECISION.getCompressionLevel());
    }

    public void testFromValueDefaultsToFullPrecision() {
        assertEquals(Encoder.QuantizationBits.FULL_PRECISION, Encoder.QuantizationBits.fromValue(999));
    }
}
