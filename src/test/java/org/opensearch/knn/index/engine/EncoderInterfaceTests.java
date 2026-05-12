/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.faiss.FaissFlatEncoder;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.index.engine.lucene.LuceneSQEncoder;

public class EncoderInterfaceTests extends KNNTestCase {

    public void testFaissFlatEncoderType() {
        FaissFlatEncoder encoder = new FaissFlatEncoder();
        assertEquals(Encoder.EncoderType.FLAT, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FULL_PRECISION));
        assertEquals(1, encoder.getSupportedBits().size());
    }

    public void testFaissSQEncoderType() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(Encoder.EncoderType.SQ, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.SIXTEEN));
        assertEquals(2, encoder.getSupportedBits().size());
    }

    public void testLuceneSQEncoderType() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(Encoder.EncoderType.SQ, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.SEVEN));
        assertEquals(2, encoder.getSupportedBits().size());
    }

    public void testQFrameBitEncoderType() {
        QFrameBitEncoder encoder = new QFrameBitEncoder();
        assertEquals(Encoder.EncoderType.BQ, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.TWO));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FOUR));
        assertEquals(3, encoder.getSupportedBits().size());
    }
}
