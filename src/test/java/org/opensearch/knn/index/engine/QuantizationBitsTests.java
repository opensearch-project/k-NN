/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
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

    public void testGetCompressionLevel_withHalfFloat_thenHalvedLadder() {
        assertEquals(CompressionLevel.x16, Encoder.QuantizationBits.ONE.getCompressionLevel(VectorDataType.HALF_FLOAT));
        assertEquals(CompressionLevel.x8, Encoder.QuantizationBits.TWO.getCompressionLevel(VectorDataType.HALF_FLOAT));
        assertEquals(CompressionLevel.x4, Encoder.QuantizationBits.FOUR.getCompressionLevel(VectorDataType.HALF_FLOAT));
    }

    public void testGetCompressionLevel_withHalfFloatAndUnsupportedBits_thenThrows() {
        for (Encoder.QuantizationBits bits : new Encoder.QuantizationBits[] {
            Encoder.QuantizationBits.SEVEN,
            Encoder.QuantizationBits.SIXTEEN,
            Encoder.QuantizationBits.FULL_PRECISION }) {
            expectThrows(IllegalArgumentException.class, () -> bits.getCompressionLevel(VectorDataType.HALF_FLOAT));
        }
    }

    public void testFromCompressionLevel_withHalfFloat_thenHalvedLadder() {
        assertEquals(
            Encoder.QuantizationBits.ONE,
            Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x16, VectorDataType.HALF_FLOAT)
        );
        assertEquals(
            Encoder.QuantizationBits.TWO,
            Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x8, VectorDataType.HALF_FLOAT)
        );
        assertEquals(
            Encoder.QuantizationBits.FOUR,
            Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x4, VectorDataType.HALF_FLOAT)
        );
    }

    public void testFromCompressionLevel_withHalfFloatAndX1_thenFullPrecision() {
        assertEquals(
            Encoder.QuantizationBits.FULL_PRECISION,
            Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x1, VectorDataType.HALF_FLOAT)
        );
    }

    public void testCompressionLevelRoundTrips_forFloatAndHalfFloat() {
        for (VectorDataType vectorDataType : new VectorDataType[] { VectorDataType.FLOAT, VectorDataType.HALF_FLOAT }) {
            java.util.Set<Encoder.QuantizationBits> supported = vectorDataType == VectorDataType.HALF_FLOAT
                ? java.util.Set.of(Encoder.QuantizationBits.ONE, Encoder.QuantizationBits.TWO, Encoder.QuantizationBits.FOUR)
                : java.util.Set.of(
                    Encoder.QuantizationBits.ONE,
                    Encoder.QuantizationBits.TWO,
                    Encoder.QuantizationBits.FOUR,
                    Encoder.QuantizationBits.SEVEN,
                    Encoder.QuantizationBits.SIXTEEN
                );
            for (Encoder.QuantizationBits bits : supported) {
                CompressionLevel level = bits.getCompressionLevel(vectorDataType);
                assertEquals(vectorDataType + " @ " + bits, bits, Encoder.QuantizationBits.fromCompressionLevel(level, vectorDataType));
            }
        }
    }

    public void testIsSQCoded_withCompressionLevel_thenFollowsEachDataTypesLadder() {
        for (CompressionLevel level : new CompressionLevel[] { CompressionLevel.x32, CompressionLevel.x16, CompressionLevel.x8 }) {
            assertTrue("FLOAT at " + level, Encoder.QuantizationBits.isSQCoded(level, VectorDataType.FLOAT));
        }
        for (CompressionLevel level : new CompressionLevel[] {
            CompressionLevel.x64,
            CompressionLevel.x4,
            CompressionLevel.x2,
            CompressionLevel.x1,
            CompressionLevel.NOT_CONFIGURED }) {
            assertFalse("FLOAT at " + level, Encoder.QuantizationBits.isSQCoded(level, VectorDataType.FLOAT));
        }
        for (CompressionLevel level : new CompressionLevel[] { CompressionLevel.x16, CompressionLevel.x8, CompressionLevel.x4 }) {
            assertTrue("HALF_FLOAT at " + level, Encoder.QuantizationBits.isSQCoded(level, VectorDataType.HALF_FLOAT));
        }
        for (CompressionLevel level : new CompressionLevel[] {
            CompressionLevel.x64,
            CompressionLevel.x32,
            CompressionLevel.x2,
            CompressionLevel.x1,
            CompressionLevel.NOT_CONFIGURED }) {
            assertFalse("HALF_FLOAT at " + level, Encoder.QuantizationBits.isSQCoded(level, VectorDataType.HALF_FLOAT));
        }
        // Types without an SQ path are never SQ-coded, even where the arithmetic would land on 1/2/4 bits.
        assertFalse(Encoder.QuantizationBits.isSQCoded(CompressionLevel.x1, VectorDataType.BINARY));
        assertFalse(Encoder.QuantizationBits.isSQCoded(CompressionLevel.x8, VectorDataType.BYTE));
    }
}
