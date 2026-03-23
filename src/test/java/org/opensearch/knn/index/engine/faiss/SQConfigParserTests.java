/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;

public class SQConfigParserTests extends KNNTestCase {

    public void testToCsv_whenValidConfig_thenReturnCsv() {
        SQConfig config = SQConfig.builder().bits(1).build();
        assertEquals("bits=1", SQConfigParser.toCsv(config));
    }

    public void testToCsv_whenBits16_thenReturnCsv() {
        SQConfig config = SQConfig.builder().bits(16).build();
        assertEquals("bits=16", SQConfigParser.toCsv(config));
    }

    public void testToCsv_whenNull_thenReturnEmpty() {
        assertEquals("", SQConfigParser.toCsv(null));
    }

    public void testToCsv_whenEmpty_thenReturnEmpty() {
        assertEquals("", SQConfigParser.toCsv(SQConfig.EMPTY));
    }

    public void testFromCsv_whenValidCsv_thenReturnConfig() {
        SQConfig config = SQConfigParser.fromCsv("bits=1");
        assertEquals(1, config.getBits());
    }

    public void testFromCsv_whenBits16_thenReturnConfig() {
        SQConfig config = SQConfigParser.fromCsv("bits=16");
        assertEquals(16, config.getBits());
    }

    public void testFromCsv_whenNull_thenReturnEmpty() {
        assertSame(SQConfig.EMPTY, SQConfigParser.fromCsv(null));
    }

    public void testFromCsv_whenEmptyString_thenReturnEmpty() {
        assertSame(SQConfig.EMPTY, SQConfigParser.fromCsv(""));
    }

    public void testFromCsv_whenInvalidFormat_thenThrow() {
        expectThrows(IllegalArgumentException.class, () -> SQConfigParser.fromCsv("invalid"));
    }

    public void testFromCsv_whenWrongKey_thenThrow() {
        expectThrows(IllegalArgumentException.class, () -> SQConfigParser.fromCsv("wrong=1"));
    }

    public void testRoundTrip() {
        SQConfig original = SQConfig.builder().bits(1).build();
        SQConfig parsed = SQConfigParser.fromCsv(SQConfigParser.toCsv(original));
        assertEquals(original, parsed);
    }
}
