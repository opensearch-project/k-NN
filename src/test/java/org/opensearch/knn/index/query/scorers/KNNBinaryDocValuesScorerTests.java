/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;

public class KNNBinaryDocValuesScorerTests extends TestCase {

    @SneakyThrows
    public void testFloatQuery_withL2SpaceType_scoresCorrectly() {
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f };
        List<float[]> docVectors = List.of(
            new float[] { 1.0f, 2.0f, 3.0f },  // identical to query
            new float[] { 4.0f, 5.0f, 6.0f },   // far from query
            new float[] { 1.1f, 2.1f, 3.1f }    // close to query
        );

        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            docVectors
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(queryVector, binaryDocValues, SpaceType.L2);

        DocIdSetIterator iterator = scorer.iterator();

        // Doc 0: identical vector, highest score
        assertEquals(0, iterator.nextDoc());
        float score0 = scorer.score();

        // Doc 1: far vector, lowest score
        assertEquals(1, iterator.nextDoc());
        float score1 = scorer.score();

        // Doc 2: close vector, middle score
        assertEquals(2, iterator.nextDoc());
        float score2 = scorer.score();

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        // Identical vector should have the highest score, far vector the lowest
        assertTrue("Identical vector should score higher than close vector", score0 > score2);
        assertTrue("Close vector should score higher than far vector", score2 > score1);
    }

    @SneakyThrows
    public void testFloatQuery_withCosineSpaceType_scoresCorrectly() {
        float[] queryVector = new float[] { 1.0f, 0.0f };
        List<float[]> docVectors = List.of(
            new float[] { 1.0f, 0.0f },   // same direction
            new float[] { 0.0f, 1.0f }    // orthogonal
        );

        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            docVectors
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(queryVector, binaryDocValues, SpaceType.COSINESIMIL);

        DocIdSetIterator iterator = scorer.iterator();

        assertEquals(0, iterator.nextDoc());
        float scoreSameDir = scorer.score();

        assertEquals(1, iterator.nextDoc());
        float scoreOrthogonal = scorer.score();

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        assertTrue("Same direction should score higher than orthogonal", scoreSameDir > scoreOrthogonal);
    }

    @SneakyThrows
    public void testByteQuery_withHammingSpaceType_scoresCorrectly() {
        byte[] queryVector = new byte[] { 0b0000_0000, (byte) 0b1111_1111 };
        List<byte[]> docVectors = List.of(
            new byte[] { 0b0000_0000, (byte) 0b1111_1111 },  // identical
            new byte[] { (byte) 0b1111_1111, 0b0000_0000 }   // all bits flipped
        );

        TestVectorValues.PredefinedByteVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedByteVectorBinaryDocValues(
            docVectors
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(queryVector, binaryDocValues, SpaceType.HAMMING);

        DocIdSetIterator iterator = scorer.iterator();

        assertEquals(0, iterator.nextDoc());
        float scoreIdentical = scorer.score();

        assertEquals(1, iterator.nextDoc());
        float scoreFlipped = scorer.score();

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        assertTrue("Identical vector should score higher than all-flipped vector", scoreIdentical > scoreFlipped);
    }

    @SneakyThrows
    public void testIterator_returnsSameBinaryDocValuesInstance() {
        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            List.of(new float[] { 1.0f })
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(new float[] { 1.0f }, binaryDocValues, SpaceType.L2);

        assertSame(binaryDocValues, scorer.iterator());
    }

    @SneakyThrows
    public void testFloatQuery_withEmptyDocValues_returnsNoMoreDocs() {
        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            List.of(new float[] { 1.0f })
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(new float[] { 1.0f }, binaryDocValues, SpaceType.L2);

        DocIdSetIterator iterator = scorer.iterator();
        assertEquals(0, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testFloatQuery_withAdvance_scoresCorrectDoc() {
        float[] queryVector = new float[] { 1.0f, 2.0f };
        List<float[]> docVectors = List.of(
            new float[] { 0.0f, 0.0f },
            new float[] { 1.0f, 2.0f },  // identical to query
            new float[] { 3.0f, 4.0f }
        );

        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            docVectors
        );
        KNNBinaryDocValuesScorer scorer = KNNBinaryDocValuesScorer.create(queryVector, binaryDocValues, SpaceType.L2);

        DocIdSetIterator iterator = scorer.iterator();

        // Skip directly to doc 1
        assertEquals(1, iterator.advance(1));
        float scoreDoc1 = scorer.score();

        // Advance to doc 2
        assertEquals(2, iterator.nextDoc());
        float scoreDoc2 = scorer.score();

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        // Doc 1 is identical to query, should score higher
        assertTrue("Doc 1 (identical) should score higher than doc 2", scoreDoc1 > scoreDoc2);
    }
}
