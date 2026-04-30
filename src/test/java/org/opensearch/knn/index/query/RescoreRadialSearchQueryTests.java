/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

// Tests for RescoreRadialSearchQuery — the pass-through skeleton.
// At this stage the wrapper delegates entirely to the inner query without rescoring.
// These tests verify the wrapper structure and delegation, not rescoring correctness.
public class RescoreRadialSearchQueryTests extends KNNTestCase {
    private static final String FIELD_NAME = "test-field";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f };
    private static final float RADIUS = 0.5f;

    // Verify scorerSupplier delegates to inner weight and returns the same results
    public void testScorerSupplier_passThrough() throws IOException {
        // Set up inner query results: 3 docs with known scores
        TopDocs innerTopDocs = new TopDocs(
            new TotalHits(3, TotalHits.Relation.EQUAL_TO),
            new ScoreDoc[] { new ScoreDoc(0, 0.9f), new ScoreDoc(1, 0.7f), new ScoreDoc(2, 0.5f) }
        );
        Scorer innerScorer = new KNNScorer(innerTopDocs, 1.0f);

        ScorerSupplier innerScorerSupplier = mock(ScorerSupplier.class);
        when(innerScorerSupplier.get(any(Long.class))).thenReturn(innerScorer);
        when(innerScorerSupplier.cost()).thenReturn(3L);

        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);
        when(innerWeight.scorerSupplier(leafContext)).thenReturn(innerScorerSupplier);

        Query innerQuery = mock(Query.class);
        when(innerQuery.rewrite(any(IndexSearcher.class))).thenReturn(innerQuery);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        ScorerSupplier supplier = weight.scorerSupplier(leafContext);

        assertNotNull(supplier);
        assertEquals(3L, supplier.cost());

        Scorer scorer = supplier.get(0);
        assertNotNull(scorer);

        // Iterate and verify all 3 docs are returned in doc ID order (TopDocsDISI sorts by docId)
        int count = 0;
        while (scorer.iterator().nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            count++;
        }
        assertEquals(3, count);
    }

    // Verify that when inner scorer supplier is null, our supplier returns null
    public void testScorerSupplier_whenInnerReturnsNull_thenReturnsNull() throws IOException {
        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);
        when(innerWeight.scorerSupplier(leafContext)).thenReturn(null);

        Query innerQuery = mock(Query.class);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        ScorerSupplier supplier = weight.scorerSupplier(leafContext);

        assertNull(supplier);
    }

    // Verify that when inner scorer returns empty TopDocs, we get an empty scorer
    public void testScorerSupplier_whenInnerReturnsEmpty_thenEmptyScorer() throws IOException {
        Scorer innerScorer = KNNScorer.emptyScorer();

        ScorerSupplier innerScorerSupplier = mock(ScorerSupplier.class);
        when(innerScorerSupplier.get(any(Long.class))).thenReturn(innerScorer);
        when(innerScorerSupplier.cost()).thenReturn(0L);

        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);
        when(innerWeight.scorerSupplier(leafContext)).thenReturn(innerScorerSupplier);

        Query innerQuery = mock(Query.class);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        ScorerSupplier supplier = weight.scorerSupplier(leafContext);

        assertNotNull(supplier);
        Scorer scorer = supplier.get(0);
        assertNotNull(scorer);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    // Verify equals/hashCode contract
    public void testEqualsAndHashCode() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery q1 = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        RescoreRadialSearchQuery q2 = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);

        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        // Different radius
        RescoreRadialSearchQuery q3 = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, 0.9f);
        assertNotEquals(q1, q3);

        // Different field
        RescoreRadialSearchQuery q4 = new RescoreRadialSearchQuery(innerQuery, "other-field", QUERY_VECTOR, RADIUS);
        assertNotEquals(q1, q4);

        // Different vector
        RescoreRadialSearchQuery q5 = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, new float[] { 9.0f }, RADIUS);
        assertNotEquals(q1, q5);
    }

    // Verify toString contains useful information
    public void testToString() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        String str = query.toString(FIELD_NAME);

        assertTrue(str.contains("RescoreRadialSearchQuery"));
        assertTrue(str.contains(FIELD_NAME));
        assertTrue(str.contains(String.valueOf(RADIUS)));
    }

    // Verify getters expose the fields correctly
    public void testGetters() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);

        assertSame(innerQuery, query.getInnerQuery());
        assertEquals(FIELD_NAME, query.getField());
        assertArrayEquals(QUERY_VECTOR, query.getQueryVector(), 0.0f);
        assertEquals(RADIUS, query.getRadius(), 0.0f);
    }
}
