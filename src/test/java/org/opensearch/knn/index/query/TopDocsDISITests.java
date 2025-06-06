/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.junit.Before;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;

public class TopDocsDISITests extends OpenSearchTestCase {

    private TopDocsDISI topDocsDISI;
    private ScoreDoc[] sampleScoreDocs;
    private float maxScore;
    private int totalHits;

    @Before
    public void setupBeforeTest() {
        // Initialize test data
        maxScore = 1.5f;
        totalHits = 5;
        sampleScoreDocs = new ScoreDoc[] {
            new ScoreDoc(0, 1.5f),
            new ScoreDoc(1, 1.2f),
            new ScoreDoc(2, 0.9f),
            new ScoreDoc(3, 0.6f),
            new ScoreDoc(4, 0.3f) };

        topDocsDISI = new TopDocsDISI(new TopDocs(new TotalHits(totalHits, TotalHits.Relation.EQUAL_TO), sampleScoreDocs));
    }

    public void testAdvance() throws Exception {
        // Test advancing to specific positions
        assertEquals(0, topDocsDISI.advance(0));  // First doc
        assertEquals(2, topDocsDISI.advance(2));  // Middle doc
        assertEquals(4, topDocsDISI.advance(4));  // Last doc
        assertEquals(TopDocsDISI.NO_MORE_DOCS, topDocsDISI.advance(5));  // Beyond last doc
    }

    public void testDocID() throws IOException {
        // Test initial state
        assertEquals(-1, topDocsDISI.docID());

        // Test after advancing
        topDocsDISI.advance(2);
        assertEquals(2, topDocsDISI.docID());
    }

    public void testNextDoc() throws Exception {
        // Test sequential advancement
        assertEquals(0, topDocsDISI.nextDoc());
        assertEquals(1, topDocsDISI.nextDoc());
        assertEquals(2, topDocsDISI.nextDoc());
        assertEquals(3, topDocsDISI.nextDoc());
        assertEquals(4, topDocsDISI.nextDoc());
        assertEquals(TopDocsDISI.NO_MORE_DOCS, topDocsDISI.nextDoc());
    }

    public void testCost() {
        // Test cost matches the number of documents
        assertEquals(totalHits, topDocsDISI.cost());
    }

    public void testGetScore() throws IOException {
        // Test score retrieval
        topDocsDISI.advance(0);
        assertEquals(1.5f, topDocsDISI.score(), 0.001f);

        topDocsDISI.advance(2);
        assertEquals(0.9f, topDocsDISI.score(), 0.001f);

        topDocsDISI.advance(4);
        assertEquals(0.3f, topDocsDISI.score(), 0.001f);
    }

    public void testEmptyTopDocs() {
        // Test behavior with empty TopDocs
        TopDocsDISI emptyDISI = new TopDocsDISI(TopDocsCollector.EMPTY_TOPDOCS);
        assertEquals(-1, emptyDISI.docID());
        assertEquals(0, emptyDISI.cost());
    }

}
