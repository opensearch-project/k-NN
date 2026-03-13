/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.opensearch.knn.KNNTestCase;

public class MuveraEncoderTests extends KNNTestCase {

    public void testGetEmbeddingSize() {
        // FDE dimension = rReps * 2^kSim * dimProj
        MuveraEncoder encoder = new MuveraEncoder(128, 4, 8, 20, 42L);
        assertEquals(20 * 16 * 8, encoder.getEmbeddingSize()); // 2560

        MuveraEncoder small = new MuveraEncoder(4, 1, 2, 2, 42L);
        assertEquals(2 * 2 * 2, small.getEmbeddingSize()); // 8
    }

    public void testDeterministicOutput() {
        // Same seed, same input -> same output
        MuveraEncoder enc1 = new MuveraEncoder(4, 1, 2, 2, 42L);
        MuveraEncoder enc2 = new MuveraEncoder(4, 1, 2, 2, 42L);

        double[][] vectors = { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 } };

        float[] fde1 = enc1.processDocument(vectors);
        float[] fde2 = enc2.processDocument(vectors);

        assertEquals(fde1.length, fde2.length);
        for (int i = 0; i < fde1.length; i++) {
            assertEquals("FDE mismatch at index " + i, fde1[i], fde2[i], 0.0f);
        }
    }

    public void testDifferentSeedProducesDifferentOutput() {
        MuveraEncoder enc1 = new MuveraEncoder(4, 1, 2, 2, 42L);
        MuveraEncoder enc2 = new MuveraEncoder(4, 1, 2, 2, 99L);

        double[][] vectors = { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 } };

        float[] fde1 = enc1.processDocument(vectors);
        float[] fde2 = enc2.processDocument(vectors);

        boolean allEqual = true;
        for (int i = 0; i < fde1.length; i++) {
            if (fde1[i] != fde2[i]) {
                allEqual = false;
                break;
            }
        }
        assertFalse("Different seeds should produce different FDE vectors", allEqual);
    }

    public void testDocumentVsQueryModeDiffers() {
        MuveraEncoder encoder = new MuveraEncoder(4, 1, 2, 2, 42L);
        double[][] vectors = { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 } };

        float[] docFde = encoder.processDocument(vectors);
        float[] queryFde = encoder.processQuery(vectors);

        assertEquals(docFde.length, queryFde.length);

        // Document and query modes should produce different results
        // because document normalizes by count and fills empty clusters
        boolean allEqual = true;
        for (int i = 0; i < docFde.length; i++) {
            if (docFde[i] != queryFde[i]) {
                allEqual = false;
                break;
            }
        }
        assertFalse("Document and query FDE should differ", allEqual);
    }

    public void testOutputDimensionMatchesExpected() {
        MuveraEncoder encoder = new MuveraEncoder(4, 1, 2, 2, 42L);
        double[][] vectors = { { 1.0, 2.0, 3.0, 4.0 } };

        float[] docFde = encoder.processDocument(vectors);
        assertEquals(encoder.getEmbeddingSize(), docFde.length);

        float[] queryFde = encoder.processQuery(vectors);
        assertEquals(encoder.getEmbeddingSize(), queryFde.length);
    }

    public void testSingleVector() {
        MuveraEncoder encoder = new MuveraEncoder(4, 1, 2, 2, 42L);
        double[][] vectors = { { 1.0, 0.0, 0.0, 0.0 } };

        float[] fde = encoder.processDocument(vectors);
        assertEquals(8, fde.length);

        // Should not be all zeros since the single vector fills all clusters
        boolean allZero = true;
        for (float v : fde) {
            if (v != 0.0f) {
                allZero = false;
                break;
            }
        }
        assertFalse("Single vector FDE should not be all zeros", allZero);
    }

    public void testInvalidDimThrows() {
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(0, 4, 8, 20, 42L));
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(-1, 4, 8, 20, 42L));
    }

    public void testInvalidKSimThrows() {
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(128, -1, 8, 20, 42L));
    }

    public void testInvalidDimProjThrows() {
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(128, 4, 0, 20, 42L));
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(128, 4, -1, 20, 42L));
    }

    public void testInvalidRRepsThrows() {
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(128, 4, 8, 0, 42L));
        expectThrows(IllegalArgumentException.class, () -> new MuveraEncoder(128, 4, 8, -1, 42L));
    }

    public void testKSimZeroMeansSinglePartition() {
        // kSim=0 means 2^0 = 1 partition, so FDE = rReps * 1 * dimProj
        MuveraEncoder encoder = new MuveraEncoder(4, 0, 2, 3, 42L);
        assertEquals(3 * 1 * 2, encoder.getEmbeddingSize()); // 6

        double[][] vectors = { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 } };
        float[] fde = encoder.processDocument(vectors);
        assertEquals(6, fde.length);
    }
}
