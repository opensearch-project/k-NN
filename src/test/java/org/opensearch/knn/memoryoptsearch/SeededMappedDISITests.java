/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.junit.Before;
import org.junit.Test;
import org.mockito.InOrder;
import org.opensearch.lucene.SeededMappedDISI;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SeededMappedDISITests {
    private KnnVectorValues.DocIndexIterator indexedDISI;
    private DocIdSetIterator sourceDISI;
    private SeededMappedDISI mappedDISI;

    @Before
    public void setup() {
        indexedDISI = mock(KnnVectorValues.DocIndexIterator.class);
        sourceDISI = mock(DocIdSetIterator.class);
        mappedDISI = new SeededMappedDISI(indexedDISI, sourceDISI);
    }

    @Test
    public void testNextDocAdvancesBothIterators() throws IOException {
        // Arrange
        when(sourceDISI.nextDoc()).thenReturn(10);
        when(indexedDISI.advance(10)).thenReturn(10);
        when(indexedDISI.docID()).thenReturn(10);
        when(sourceDISI.docID()).thenReturn(10);
        when(indexedDISI.index()).thenReturn(42); // vector index

        // Act
        int result = mappedDISI.nextDoc();

        // Assert
        verify(sourceDISI).nextDoc();
        verify(indexedDISI).advance(10);
        assertEquals("Should return vector index mapped to docID", 42, result);
    }

    @Test
    public void testNextDocNoMoreDocs() throws IOException {
        when(sourceDISI.nextDoc()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(indexedDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(sourceDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        int result = mappedDISI.nextDoc();

        assertEquals("Should return NO_MORE_DOCS when exhausted", DocIdSetIterator.NO_MORE_DOCS, result);
        verify(indexedDISI, never()).advance(anyInt());
    }

    @Test
    public void testAdvanceSyncsIndexedDISI() throws IOException {
        when(sourceDISI.advance(25)).thenReturn(25);
        when(indexedDISI.advance(25)).thenReturn(25);
        when(indexedDISI.docID()).thenReturn(25);
        when(sourceDISI.docID()).thenReturn(25);
        when(indexedDISI.index()).thenReturn(7);

        int result = mappedDISI.advance(25);

        InOrder order = inOrder(sourceDISI, indexedDISI);
        order.verify(sourceDISI).advance(25);
        order.verify(indexedDISI).advance(25);

        assertEquals("Should return mapped index for target doc", 7, result);
    }

    @Test
    public void testAdvanceNoMoreDocs() throws IOException {
        when(sourceDISI.advance(100)).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(indexedDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(sourceDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        int result = mappedDISI.advance(100);

        assertEquals("Should return NO_MORE_DOCS when exhausted", DocIdSetIterator.NO_MORE_DOCS, result);
        verify(indexedDISI, never()).advance(anyInt());
    }

    @Test
    public void testDocIDReturnsVectorIndex() {
        when(indexedDISI.docID()).thenReturn(10);
        when(sourceDISI.docID()).thenReturn(10);
        when(indexedDISI.index()).thenReturn(99);

        int docID = mappedDISI.docID();

        assertEquals("Should return mapped vector index", 99, docID);
    }

    @Test
    public void testDocIDReturnsNoMoreDocsIfEitherIteratorExhausted() {
        when(indexedDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        when(sourceDISI.docID()).thenReturn(5);
        assertEquals("If indexedDISI exhausted, return NO_MORE_DOCS", DocIdSetIterator.NO_MORE_DOCS, mappedDISI.docID());

        when(indexedDISI.docID()).thenReturn(3);
        when(sourceDISI.docID()).thenReturn(DocIdSetIterator.NO_MORE_DOCS);
        assertEquals("If sourceDISI exhausted, return NO_MORE_DOCS", DocIdSetIterator.NO_MORE_DOCS, mappedDISI.docID());
    }

    @Test
    public void testCostDelegatesToSource() {
        when(sourceDISI.cost()).thenReturn(123L);
        assertEquals("Cost should delegate to sourceDISI", 123L, mappedDISI.cost());
        verify(sourceDISI).cost();
    }

    @Test
    public void testSequentialNextDocCallsAdvanceInOrder() throws IOException {
        when(sourceDISI.nextDoc()).thenReturn(5).thenReturn(8).thenReturn(DocIdSetIterator.NO_MORE_DOCS);

        when(indexedDISI.advance(5)).thenReturn(5);
        when(indexedDISI.advance(8)).thenReturn(8);
        when(indexedDISI.docID()).thenReturn(8);
        when(sourceDISI.docID()).thenReturn(8);
        when(indexedDISI.index()).thenReturn(33);

        mappedDISI.nextDoc(); // doc 5
        mappedDISI.nextDoc(); // doc 8
        mappedDISI.nextDoc(); // end

        InOrder inOrder = inOrder(sourceDISI, indexedDISI);
        inOrder.verify(sourceDISI).nextDoc();
        inOrder.verify(indexedDISI).advance(5);
        inOrder.verify(sourceDISI).nextDoc();
        inOrder.verify(indexedDISI).advance(8);
    }
}
