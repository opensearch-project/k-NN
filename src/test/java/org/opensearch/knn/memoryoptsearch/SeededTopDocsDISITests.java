/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.junit.Test;
import org.opensearch.lucene.SeededTopDocsDISI;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class SeededTopDocsDISITests {
    private static TopDocs topDocs(int... docIds) {
        // Make score docs having scores [1,2,3,...]
        ScoreDoc[] scoreDocs = new ScoreDoc[docIds.length];
        for (int i = 0; i < docIds.length; i++) {
            scoreDocs[i] = new ScoreDoc(docIds[i], i);
        }
        return new TopDocs(new TotalHits(docIds.length, TotalHits.Relation.EQUAL_TO), scoreDocs);
    }

    @Test
    public void testSortedOrderAfterConstruction() {
        TopDocs unsorted = topDocs(5, 2, 9, 1);
        SeededTopDocsDISI disi = new SeededTopDocsDISI(unsorted);

        // Force iteration to confirm sorting
        int doc;
        int last = -1;
        while ((doc = disi.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            assertTrue("Doc IDs should be in ascending order", doc > last);
            last = doc;
        }
    }

    @Test
    public void testDocIDBeforeIteration() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(1, 2, 3));
        assertEquals("docID should be -1 before iteration starts", -1, disi.docID());
    }

    @Test
    public void testNextDocIteratesAll() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(10, 5, 7));
        int[] expected = { 5, 7, 10 };
        int i = 0;
        int doc;
        while ((doc = disi.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            assertEquals(expected[i++], doc);
        }
        assertEquals("Should iterate over all docs", expected.length, i);
    }

    @Test
    public void testDocIDAfterExhaustion() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(1));
        disi.nextDoc(); // first doc
        disi.nextDoc(); // should hit NO_MORE_DOCS
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, disi.docID());
    }

    @Test
    public void testCostMatchesDocCount() {
        TopDocs docs = topDocs(3, 1, 2);
        SeededTopDocsDISI disi = new SeededTopDocsDISI(docs);
        assertEquals(docs.scoreDocs.length, disi.cost());
    }

    @Test
    @SneakyThrows
    public void testAdvanceToTarget() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(2, 5, 7, 10));
        int advanced = disi.advance(6);
        assertEquals(7, advanced);
        assertEquals(7, disi.docID());
    }

    @Test
    @SneakyThrows
    public void testAdvancePastEndReturnsNoMoreDocs() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(1, 2, 3));
        int doc = disi.advance(10);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, doc);
    }

    @Test
    @SneakyThrows
    public void testEmptyTopDocs() {
        TopDocs empty = new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        SeededTopDocsDISI disi = new SeededTopDocsDISI(empty);

        assertEquals(0, disi.cost());
        assertEquals(-1, disi.docID());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, disi.nextDoc());
        assertThrows(AssertionError.class, () -> disi.advance(0));
    }

    @Test
    @SneakyThrows
    public void testSequentialNextDocThenAdvance() {
        SeededTopDocsDISI disi = new SeededTopDocsDISI(topDocs(1, 4, 9, 15));

        // Move to second doc
        assertEquals(1, disi.nextDoc());
        assertEquals(4, disi.nextDoc());

        // Now advance beyond 4 to 10
        assertEquals(15, disi.advance(10));
        assertEquals(15, disi.docID());

        // Advance beyond last
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, disi.advance(16));
    }
}
