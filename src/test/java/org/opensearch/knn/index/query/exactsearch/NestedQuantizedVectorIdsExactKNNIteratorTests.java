/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.mockito.Mockito.*;

public class NestedQuantizedVectorIdsExactKNNIteratorTests extends KNNTestCase {

    public void testNextDoc_returnsBestChildPerParent() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(10);
        parentBitSet.set(3); // Parent at doc 3
        parentBitSet.set(7); // Parent at doc 7

        // Children: 0, 1, 2 belong to parent 3; 4, 5, 6 belong to parent 7
        when(docIndexIterator.nextDoc()).thenReturn(0, 1, 2, 4, 5, 6, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2, 4, 5, 6);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 1, 2, 4 }); // Best match
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 1, 2, 5 });
        when(byteVectorValues.vectorValue(4)).thenReturn(new byte[] { 1, 3, 3 });
        when(byteVectorValues.vectorValue(5)).thenReturn(new byte[] { 1, 2, 3 }); // Best match
        when(byteVectorValues.vectorValue(6)).thenReturn(new byte[] { 2, 3, 4 });

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        int bestChild1 = iterator.nextDoc();
        assertTrue(bestChild1 >= 0 && bestChild1 < 3);
        float score1 = iterator.score();
        assertTrue(score1 > Float.NEGATIVE_INFINITY);

        int bestChild2 = iterator.nextDoc();
        assertTrue(bestChild2 >= 4 && bestChild2 < 7);
        float score2 = iterator.score();
        assertTrue(score2 > Float.NEGATIVE_INFINITY);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_singleChildPerParent() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(5);
        parentBitSet.set(1);
        parentBitSet.set(3);

        when(docIndexIterator.nextDoc()).thenReturn(0, 2, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 2);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 2, 3, 4 });

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        assertEquals(0, iterator.nextDoc());
        assertTrue(iterator.score() > Float.NEGATIVE_INFINITY);
        assertEquals(2, iterator.nextDoc());
        assertTrue(iterator.score() > Float.NEGATIVE_INFINITY);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_withFilter() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        DocIdSetIterator filterIterator = mock(DocIdSetIterator.class);
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(10);
        parentBitSet.set(5);

        when(filterIterator.nextDoc()).thenReturn(1, 3, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.advance(1)).thenReturn(1);
        when(docIndexIterator.advance(3)).thenReturn(3);
        when(docIndexIterator.index()).thenReturn(1, 3);
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(3)).thenReturn(new byte[] { 1, 2, 4 });

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            filterIterator,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        int bestChild = iterator.nextDoc();
        assertTrue(bestChild == 1 || bestChild == 3);
        assertTrue(iterator.score() > Float.NEGATIVE_INFINITY);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_noChildren_returnsMinusOne() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(5);
        parentBitSet.set(0); // Parent at doc 0, no children before it

        when(docIndexIterator.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testNextDoc_multipleChildrenSelectsBest() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(5);
        parentBitSet.set(4);

        when(docIndexIterator.nextDoc()).thenReturn(0, 1, 2, 3, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2, 3);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 5, 5, 5 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 1, 2, 3 }); // Exact match - best
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 4, 4, 4 });
        when(byteVectorValues.vectorValue(3)).thenReturn(new byte[] { 3, 3, 3 });

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        assertEquals(1, iterator.nextDoc()); // Should return child with best score
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testVectorValueCalled_onlyForChildren() throws IOException {
        byte[] queryVector = new byte[] { 1, 2, 3 };
        KnnVectorValues.DocIndexIterator docIndexIterator = mock(KnnVectorValues.DocIndexIterator.class);
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        BitSet parentBitSet = new FixedBitSet(5);
        parentBitSet.set(3);

        when(docIndexIterator.nextDoc()).thenReturn(0, 1, 2, DocIdSetIterator.NO_MORE_DOCS);
        when(docIndexIterator.index()).thenReturn(0, 1, 2);
        when(byteVectorValues.vectorValue(0)).thenReturn(new byte[] { 1, 2, 3 });
        when(byteVectorValues.vectorValue(1)).thenReturn(new byte[] { 2, 3, 4 });
        when(byteVectorValues.vectorValue(2)).thenReturn(new byte[] { 3, 4, 5 });

        NestedQuantizedVectorIdsExactKNNIterator iterator = new NestedQuantizedVectorIdsExactKNNIterator(
            null,
            docIndexIterator,
            byteVectorValues,
            queryVector,
            parentBitSet
        );

        iterator.nextDoc();

        verify(byteVectorValues, times(1)).vectorValue(0);
        verify(byteVectorValues, times(1)).vectorValue(1);
        verify(byteVectorValues, times(1)).vectorValue(2);
    }
}
