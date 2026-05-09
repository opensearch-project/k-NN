/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.QueryVisitor;
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

    // Given: a RescoreRadialSearchQuery with an inner query that rewrites to a different query
    // When: rewrite() is called
    // Then: a new RescoreRadialSearchQuery is returned wrapping the rewritten inner query
    // When: rewrite() is called again on the result
    // Then: it converges — returns the same instance (this) since inner query no longer changes
    public void testRewrite_converges() throws IOException {
        // Given: inner query that rewrites to a different query on first call
        Query originalInner = mock(Query.class);
        Query rewrittenInner = new MatchAllDocsQuery();
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(originalInner.rewrite(searcher)).thenReturn(rewrittenInner);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(originalInner, FIELD_NAME, QUERY_VECTOR, RADIUS);

        // When: first rewrite — inner query changes, so a new wrapper is created
        Query firstRewrite = query.rewrite(searcher);

        // Then: result is a different instance wrapping the rewritten inner query
        assertTrue(firstRewrite instanceof RescoreRadialSearchQuery);
        assertNotSame(query, firstRewrite);
        RescoreRadialSearchQuery rewrittenQuery = (RescoreRadialSearchQuery) firstRewrite;
        assertSame(rewrittenInner, rewrittenQuery.getInnerQuery());
        assertEquals(FIELD_NAME, rewrittenQuery.getField());
        assertEquals(RADIUS, rewrittenQuery.getRadius(), 0.0f);

        // When: second rewrite — inner query (MatchAllDocsQuery) rewrites to itself
        Query secondRewrite = firstRewrite.rewrite(searcher);

        // Then: converges — returns the same instance since inner didn't change
        assertSame(firstRewrite, secondRewrite);
    }

    // Given: a RescoreRadialSearchQuery whose inner query already rewrites to itself
    // When: rewrite() is called
    // Then: returns the same instance (this) immediately — no new object created
    public void testRewrite_whenInnerAlreadyRewritten_thenReturnsSameInstance() throws IOException {
        Query innerQuery = new MatchAllDocsQuery();
        IndexSearcher searcher = mock(IndexSearcher.class);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Query rewritten = query.rewrite(searcher);

        // Returns this — same instance, not just equal
        assertSame(query, rewritten);
    }

    // Given: a RescoreRadialSearchQuery wrapping an inner query
    // When: visit() is called
    // Then: the visitor is propagated to the inner query via getSubVisitor(MUST),
    // allowing tools (highlighting, field analysis) to discover the inner query
    public void testVisit() {
        // Given: a RescoreRadialSearchQuery wrapping a MatchAllDocsQuery
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        final boolean[] innerVisited = { false };

        // When: visit() is called — it propagates to the inner query via getSubVisitor(MUST)
        query.visit(new QueryVisitor() {
            @Override
            public void visitLeaf(Query q) {
                innerVisited[0] = true;
                // MatchAllDocsQuery.visit() calls visitLeaf(this) on the sub-visitor,
                // which reaches our visitor since the default getSubVisitor returns itself
                assertSame(innerQuery, q);
            }
        });

        // Then: the inner query was visited through the sub-visitor chain
        assertTrue("visit() should propagate to the inner query via getSubVisitor", innerVisited[0]);
    }

    // RescoreWeight.explain() delegates to the inner weight's explain because the explanation
    // should reflect the inner query's scoring logic (quantized radial search). Once rescoring
    // is added, this may be enhanced to include rescore details.
    public void testExplain() throws IOException {
        // Given: an inner weight that returns a known explanation for doc 42
        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);
        Explanation expectedExplanation = Explanation.noMatch("test explanation");
        when(innerWeight.explain(leafContext, 42)).thenReturn(expectedExplanation);

        Query innerQuery = mock(Query.class);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        // When: explain is called on the RescoreWeight for doc 42
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        Explanation explanation = weight.explain(leafContext, 42);

        // Then: the explanation is the same object returned by the inner weight (pass-through)
        assertSame(expectedExplanation, explanation);
    }

    // The rescore result is deterministic for the same query parameters and segment state,
    // so it is safe for Lucene's query cache to cache the results.
    public void testIsCacheable() throws IOException {
        // Given: a RescoreWeight created from a mock inner weight
        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);

        Query innerQuery = mock(Query.class);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS);
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);

        // When/Then: isCacheable returns true for any leaf context
        assertTrue("RescoreWeight should be cacheable", weight.isCacheable(leafContext));
    }
}
