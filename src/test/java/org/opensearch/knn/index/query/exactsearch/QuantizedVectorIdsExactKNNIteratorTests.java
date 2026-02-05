/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class QuantizedVectorIdsExactKNNIteratorTests extends KNNTestCase {

    public void testNextDoc_whenNoFilter_thenIteratesAllDocs() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(0, 1, 2, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 3, 4, 5 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(0, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(1, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(2, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_withFilter_thenIteratesFilteredDocs() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        DocIdSetIterator filterIterator = mock(DocIdSetIterator.class);
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(filterIterator.nextDoc()).thenReturn(1, 3, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.advance(1)).thenReturn(1);
        when(docIndexIterator.advance(3)).thenReturn(3);
        when(docIndexIterator.index()).thenReturn(1, 3);
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(3)).thenReturn(new byte[] { 4, 5, 6 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            filterIterator,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(1, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(3, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testComputeScore_usesHammingDistance() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(0, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        iterator.nextDoc();
        float score = iterator.score();

        // Hamming distance for identical vectors should give max score
        float expectedScore = SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(queryVector, queryVector);
        assertEquals(expectedScore, score, 0.001f);
    }

    public void testScore_beforeNextDoc_returnsNegativeInfinity() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(0);

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(Float.NEGATIVE_INFINITY, iterator.score(), 0.01f);
    }

    public void testNextDoc_whenNoMoreDocs_returnsNoMoreDocs() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_multipleCallsAfterEnd_returnsNoMoreDocs() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(0, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(0, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testComputeScore_differentVectors_returnsDifferentScores() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(0, 1, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 4, 5, 6 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        iterator.nextDoc();
        float score1 = iterator.score();
        iterator.nextDoc();
        float score2 = iterator.score();

        assertNotEquals(score1, score2);
    }

    public void testWithFilter_emptyFilter_returnsNoMoreDocs() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        DocIdSetIterator filterIterator = mock(DocIdSetIterator.class);
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(filterIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            filterIterator,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testOrdinalMapping_usesIndexNotDocId() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        // Doc IDs: 5, 10, 15 but ordinals: 0, 1, 2
        when(docIndexIterator.nextDoc()).thenReturn(5, 10, 15, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 3, 4, 5 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        assertEquals(5, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(10, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
        assertEquals(15, iterator.nextDoc());
        assertTrue(iterator.score() >= 0);
    }

    public void testVectorValueCalled_verifyCallCount() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(docIndexIterator.nextDoc()).thenReturn(1, 3, 5, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 3, 4, 5 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        iterator.nextDoc();
        iterator.nextDoc();
        iterator.nextDoc();

        verify(byteVectorValues, times(1)).vectorValue(0);
        verify(byteVectorValues, times(1)).vectorValue(1);
        verify(byteVectorValues, times(1)).vectorValue(2);
    }

    public void testVectorValueCalled_withFilter_verifyCallCount() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        DocIdSetIterator filterIterator = mock(DocIdSetIterator.class);
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);

        when(filterIterator.nextDoc()).thenReturn(1, 3, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.advance(1)).thenReturn(1);
        when(docIndexIterator.advance(3)).thenReturn(3);
        when(docIndexIterator.index()).thenReturn(1, 3);
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(3)).thenReturn(new byte[] { 4, 5, 6 });

        QuantizedVectorIdsExactKNNIterator iterator = new QuantizedVectorIdsExactKNNIterator(
            filterIterator,
            docIndexIterator,
            byteVectorValues,
            queryVector
        );

        iterator.nextDoc();
        iterator.nextDoc();

        verify(byteVectorValues, times(1)).vectorValue(1);
        verify(byteVectorValues, times(1)).vectorValue(3);
        verify(byteVectorValues, times(0)).vectorValue(0);
        verify(byteVectorValues, times(0)).vectorValue(2);
    }

}
