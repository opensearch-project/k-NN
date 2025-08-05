/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.iterators.KNNIterator;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;

public class KNNLazyScorerTests extends KNNTestCase {

    public void testIteratorNextDoc() throws IOException {
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.nextDoc()).thenReturn(1, 2, DocIdSetIterator.NO_MORE_DOCS);

        KNNLazyScorer scorer = new KNNLazyScorer(mockIterator, 1.0f);
        DocIdSetIterator iterator = scorer.iterator();

        assertEquals(1, iterator.nextDoc());
        assertEquals(2, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        verify(mockIterator, times(3)).nextDoc();
    }

    public void testScoreWithBoost() throws IOException {
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.score()).thenReturn(0.8f);

        KNNLazyScorer scorer = new KNNLazyScorer(mockIterator, 2.0f);

        assertEquals(1.6f, scorer.score(), 0.0f);
        verify(mockIterator, times(1)).score();
    }

    public void testDocID() throws IOException {
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.nextDoc()).thenReturn(5);

        KNNLazyScorer scorer = new KNNLazyScorer(mockIterator, 1.0f);
        DocIdSetIterator iterator = scorer.iterator();
        iterator.nextDoc();

        assertEquals(5, scorer.docID());
    }

    public void testIteratorAdvance() throws IOException {
        KNNIterator mockIterator = mock(KNNIterator.class);
        when(mockIterator.nextDoc()).thenReturn(3, 7, DocIdSetIterator.NO_MORE_DOCS);

        KNNLazyScorer scorer = new KNNLazyScorer(mockIterator, 1.0f);
        DocIdSetIterator iterator = scorer.iterator();

        assertEquals(7, iterator.advance(5));
        verify(mockIterator, times(2)).nextDoc();
    }

    public void testGetMaxScore() throws IOException {
        KNNIterator mockIterator = mock(KNNIterator.class);
        KNNLazyScorer scorer = new KNNLazyScorer(mockIterator, 1.0f);
        assertEquals(Float.MAX_VALUE, scorer.getMaxScore(100), 0.0f);
    }
}
