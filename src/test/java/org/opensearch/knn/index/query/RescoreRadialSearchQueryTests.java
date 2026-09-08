/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.MAX_RESULTS_RADIAL_RESCORING;

// Tests for RescoreRadialSearchQuery — the pass-through skeleton.
// At this stage the wrapper delegates entirely to the inner query without rescoring.
// These tests verify the wrapper structure and delegation, not rescoring correctness.
// The quantized radial feature is disabled (#3452), so no production query path constructs this
// query today; these unit tests keep the retained implementation from silently rotting until
// the feature is re-enabled.
public class RescoreRadialSearchQueryTests extends KNNTestCase {
    private static final String FIELD_NAME = "test-field";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f };
    private static final float RADIUS = 0.5f;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        // Initialize the singleton ExactSearcher used by RescoreRadialSearchQuery
        RescoreRadialSearchQuery.initialize(new ExactSearcher(mock(ModelDao.OpenSearchKNNModelDao.class)));
    }

    // Note: the full rescoring flow (inner scorer → collectTopDocs → ExactSearcher → KNNScorer)
    // requires a real SegmentReader and cannot be unit-tested with mocks alone.
    // The rescoring-correctness integration tests (FaissSQRadialSearchIT, LuceneSQRadialSearchIT)
    // are @AwaitsFix'd while quantized radial search is disabled (#3452).

    // Given: ExactSearcher singleton is not initialized (null)
    // When: RescoreRadialSearchQuery is constructed
    // Then: NullPointerException is thrown with message about initialization
    public void testConstructor_whenExactSearcherNotInitialized_thenThrows() {
        // Temporarily set singleton to null
        RescoreRadialSearchQuery.initialize(null);
        try {
            NullPointerException e = expectThrows(
                NullPointerException.class,
                () -> new RescoreRadialSearchQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    QUERY_VECTOR,
                    RADIUS,
                    false,
                    MAX_RESULTS_RADIAL_RESCORING
                )
            );
            assertTrue(e.getMessage().contains("Exact searcher was not initialized"));
        } finally {
            // Restore for other tests
            RescoreRadialSearchQuery.initialize(new ExactSearcher(mock(ModelDao.OpenSearchKNNModelDao.class)));
        }
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

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        ScorerSupplier supplier = weight.scorerSupplier(leafContext);

        assertNull(supplier);
    }

    // Verify that when inner scorer returns empty TopDocs, we get an empty scorer
    public void testScorerSupplier_whenInnerReturnsEmpty_thenEmptyScorer() throws IOException {
        // Inner query returns an empty scorer
        Scorer innerScorer = KNNScorer.emptyScorer();

        ScorerSupplier innerScorerSupplier = mock(ScorerSupplier.class);
        when(innerScorerSupplier.get(any(Long.class))).thenReturn(innerScorer);
        when(innerScorerSupplier.cost()).thenReturn(0L);

        // Mocking inner weight
        Weight innerWeight = mock(Weight.class);
        LeafReaderContext leafContext = mock(LeafReaderContext.class);
        when(innerWeight.scorerSupplier(leafContext)).thenReturn(innerScorerSupplier);

        // IndexSearcher
        Query innerQuery = mock(Query.class);
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(searcher.rewrite(innerQuery)).thenReturn(innerQuery);
        when(searcher.createWeight(eq(innerQuery), any(ScoreMode.class), anyFloat())).thenReturn(innerWeight);

        // Set up RescoreRadialSearchQuery
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);
        ScorerSupplier supplier = weight.scorerSupplier(leafContext);

        // Validate empty iterator
        assertNotNull(supplier);
        Scorer scorer = supplier.get(0);
        assertNotNull(scorer);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    // Verify equals/hashCode contract
    public void testEqualsAndHashCode() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery q1 = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        RescoreRadialSearchQuery q2 = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );

        assertEquals(q1, q2);
        assertEquals(q1.hashCode(), q2.hashCode());

        // Different radius
        RescoreRadialSearchQuery q3 = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            0.9f,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        assertNotEquals(q1, q3);

        // Different field
        RescoreRadialSearchQuery q4 = new RescoreRadialSearchQuery(
            innerQuery,
            "other-field",
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        assertNotEquals(q1, q4);

        // Different vector
        RescoreRadialSearchQuery q5 = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            new float[] { 9.0f },
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        assertNotEquals(q1, q5);
    }

    // Verify toString contains useful information
    public void testToString() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        String str = query.toString(FIELD_NAME);

        assertTrue(str.contains("RescoreRadialSearchQuery"));
        assertTrue(str.contains(FIELD_NAME));
        assertTrue(str.contains(String.valueOf(RADIUS)));
    }

    // Given: a boost other than 1.0
    // When: the rescored scorer is wrapped in BoostedScorer
    // Then: scores and max scores are scaled, iteration is delegated, and the competitive-score
    // threshold is translated back into the delegate's own scale
    @SneakyThrows
    public void testBoostedScorer_scalesScoresAndDelegatesIteration() {
        final Scorer delegate = mock(Scorer.class);
        final DocIdSetIterator iterator = mock(DocIdSetIterator.class);
        when(delegate.docID()).thenReturn(7);
        when(delegate.iterator()).thenReturn(iterator);
        when(delegate.score()).thenReturn(0.25f);
        when(delegate.getMaxScore(100)).thenReturn(0.5f);
        when(delegate.advanceShallow(42)).thenReturn(43);

        final RescoreRadialSearchQuery.BoostedScorer boosted = new RescoreRadialSearchQuery.BoostedScorer(delegate, 4.0f);

        assertEquals(7, boosted.docID());
        assertSame(iterator, boosted.iterator());
        assertEquals(1.0f, boosted.score(), 0.0f);
        assertEquals(2.0f, boosted.getMaxScore(100), 0.0f);
        assertEquals(43, boosted.advanceShallow(42));

        // A boosted threshold of 2.0 corresponds to 0.5 in the delegate's scale.
        boosted.setMinCompetitiveScore(2.0f);
        verify(delegate).setMinCompetitiveScore(0.5f);
    }

    // Guards against divide-by-zero when translating the competitive score for a zero boost
    @SneakyThrows
    public void testBoostedScorer_whenBoostIsZero_thenMinCompetitiveScoreIsZero() {
        final Scorer delegate = mock(Scorer.class);
        final RescoreRadialSearchQuery.BoostedScorer boosted = new RescoreRadialSearchQuery.BoostedScorer(delegate, 0.0f);

        boosted.setMinCompetitiveScore(5.0f);

        verify(delegate).setMinCompetitiveScore(0.0f);
    }

    // Given: a radial query wrapped in a BoostQuery
    // When: searched over a real index
    // Then: every score is the unboosted score scaled by the boost, proving BoostedScorer is wired in
    @SneakyThrows
    public void testRescore_withBoost_thenScoresAreScaled() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = { { 1.0f, 0.0f, 0.0f }, { 0.99f, 0.01f, 0.0f }, { 0.95f, 0.05f, 0.0f } };
        final float radiusThreshold = 1.0f;
        final float boost = 3.0f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                final IndexSearcher searcher = new IndexSearcher(reader);
                final RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    queryVector,
                    radiusThreshold,
                    false,
                    vectors.length
                );

                final TopDocs plain = searcher.search(rescoreQuery, vectors.length);
                final TopDocs boosted = searcher.search(new BoostQuery(rescoreQuery, boost), vectors.length);

                assertEquals(plain.scoreDocs.length, boosted.scoreDocs.length);
                assertTrue("expected at least one hit", plain.scoreDocs.length > 0);
                for (int i = 0; i < plain.scoreDocs.length; i++) {
                    assertEquals(plain.scoreDocs[i].doc, boosted.scoreDocs[i].doc);
                    assertEquals(plain.scoreDocs[i].score * boost, boosted.scoreDocs[i].score, 1e-5f);
                }
            }
        }
    }

    // Verify getters expose the fields correctly
    public void testGetters() {
        Query innerQuery = new MatchAllDocsQuery();
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(innerQuery, FIELD_NAME, QUERY_VECTOR, RADIUS, false, 25);

        assertSame(innerQuery, query.getInnerQuery());
        assertEquals(FIELD_NAME, query.getField());
        assertArrayEquals(QUERY_VECTOR, query.getQueryVector(), 0.0f);
        assertEquals(RADIUS, query.getRadius(), 0.0f);
        assertEquals(25, query.getFirstPassK());
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

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            originalInner,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );

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

    public void testRewrite_preservesIndependentCandidateAndFinalResultLimits() throws IOException {
        Query originalInner = mock(Query.class);
        Query rewrittenInner = new MatchAllDocsQuery();
        IndexSearcher searcher = mock(IndexSearcher.class);
        when(originalInner.rewrite(searcher)).thenReturn(rewrittenInner);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(originalInner, FIELD_NAME, QUERY_VECTOR, RADIUS, false, 6);

        RescoreRadialSearchQuery rewrittenQuery = (RescoreRadialSearchQuery) query.rewrite(searcher);

        assertSame(rewrittenInner, rewrittenQuery.getInnerQuery());
        assertEquals(6, rewrittenQuery.getFirstPassK());
    }

    // Given: a RescoreRadialSearchQuery whose inner query already rewrites to itself
    // When: rewrite() is called
    // Then: returns the same instance (this) immediately — no new object created
    public void testRewrite_whenInnerAlreadyRewritten_thenReturnsSameInstance() throws IOException {
        Query innerQuery = new MatchAllDocsQuery();
        IndexSearcher searcher = mock(IndexSearcher.class);

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
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
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
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
        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
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

        RescoreRadialSearchQuery query = new RescoreRadialSearchQuery(
            innerQuery,
            FIELD_NAME,
            QUERY_VECTOR,
            RADIUS,
            false,
            MAX_RESULTS_RADIAL_RESCORING
        );
        Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE, 1.0f);

        // When/Then: isCacheable returns true for any leaf context
        assertTrue("RescoreWeight should be cacheable", weight.isCacheable(leafContext));
    }

    // --- Full rescoring flow tests using real Lucene index ---
    // These tests use MatchAllDocsQuery as the inner query so that ALL vectors pass the first phase
    // (simulating a quantized first pass that returns false positives). The rescore phase must then
    // filter out vectors whose true full-precision score falls outside the radius.

    // Given: 5 vectors indexed, all returned by inner query (first phase includes false positives)
    // When: RescoreRadialSearchQuery rescores with full precision
    // Then: only vectors within the true radius are kept, false positives are excluded
    @SneakyThrows
    public void testRescore_withLuceneIndex_filtersOutsideRadius() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 1.0f, 0.0f, 0.0f },   // identical to query, distance=0, similarity=1.0
            { 0.9f, 0.1f, 0.0f },   // very close, similarity ≈ 0.98
            { 0.8f, 0.2f, 0.0f },   // close, similarity ≈ 0.93
            { 0.0f, 1.0f, 0.0f },   // far (orthogonal), similarity ≈ 0.33 — FALSE POSITIVE
            { -1.0f, 0.0f, 0.0f },  // very far (opposite), similarity = 0.2 — FALSE POSITIVE
        };

        // Tight radius: only first 3 vectors should survive rescoring
        final float radiusThreshold = 0.9f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);

                    // Inner query returns ALL docs — simulates quantized first pass with false positives
                    Query innerQuery = new MatchAllDocsQuery();

                    // Rescore with tight radius — should filter out the false positives
                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, 10);

                    // Then: false positives (orthogonal, opposite) are excluded by rescoring
                    assertTrue("Expected some results within radius", results.scoreDocs.length > 0);
                    assertTrue(
                        "Expected fewer results than total docs (false positives filtered)",
                        results.scoreDocs.length < vectors.length
                    );
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        assertTrue(
                            "All returned docs should have score >= radiusThreshold, but got " + scoreDoc.score,
                            scoreDoc.score >= radiusThreshold
                        );
                    }
                }
            }
        }
    }

    // Given: all vectors are outside the radius, but inner query returns all of them (all are false positives)
    // When: RescoreRadialSearchQuery rescores with full precision
    // Then: all are filtered out, empty result
    @SneakyThrows
    public void testRescore_withLuceneIndex_whenAllAreFalsePositives_thenEmpty() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 0.0f, 1.0f, 0.0f },   // orthogonal, similarity ≈ 0.33
            { -1.0f, 0.0f, 0.0f },  // opposite, similarity = 0.2
            { 0.0f, 0.0f, 1.0f },   // orthogonal, similarity ≈ 0.33
        };

        // Very tight radius — none should survive
        final float radiusThreshold = 0.99f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);

                    // Inner query returns ALL docs — all are false positives
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, 10);

                    // Then: all filtered out since none are within radius
                    assertEquals("All false positives should be filtered out", 0, results.scoreDocs.length);
                }
            }
        }
    }

    // Given: all vectors are within the radius, inner query returns all (no false positives)
    // When: RescoreRadialSearchQuery rescores with full precision
    // Then: all docs kept — rescoring confirms they are all true positives
    @SneakyThrows
    public void testRescore_withLuceneIndex_whenAllWithinRadius_thenAllKept() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 1.0f, 0.0f, 0.0f },   // identical, similarity = 1.0
            { 0.99f, 0.01f, 0.0f }, // very close, similarity ≈ 0.9998
            { 0.95f, 0.05f, 0.0f }, // close, similarity ≈ 0.995
        };

        // Loose radius — all should survive
        final float radiusThreshold = 0.5f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);

                    // Inner query returns ALL docs — none are false positives
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, 10);

                    // Then: all docs kept since all are true positives
                    assertEquals("All true positives should be kept", vectors.length, results.scoreDocs.length);
                }
            }
        }
    }

    // Given: vectors indexed with COSINE similarity, inner query returns all (includes false positives)
    // When: RescoreRadialSearchQuery rescores with full precision
    // Then: only vectors with cosine similarity >= radius are kept
    @SneakyThrows
    public void testRescore_withCosine_filtersOutsideRadius() {
        // Normalized query vector for cosine
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 1.0f, 0.0f, 0.0f },   // cos=1.0, Lucene score = (1+1)/2 = 1.0
            { 0.9f, 0.1f, 0.0f },   // cos≈0.994, score ≈ 0.997
            { 0.0f, 1.0f, 0.0f },   // cos=0, score = 0.5 — FALSE POSITIVE
            { -1.0f, 0.0f, 0.0f },  // cos=-1, score = 0.0 — FALSE POSITIVE
        };

        // Lucene cosine score = (1 + cosine) / 2
        // Threshold 0.9 → only vectors with cosine > 0.8 pass
        final float radiusThreshold = 0.9f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.COSINE));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, 10);

                    // Then: false positives (orthogonal, opposite) are filtered
                    assertTrue("Expected some results", results.scoreDocs.length > 0);
                    assertTrue("Expected fewer than total (false positives removed)", results.scoreDocs.length < vectors.length);
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        assertTrue("Score should be >= threshold, got " + scoreDoc.score, scoreDoc.score >= radiusThreshold);
                    }
                }
            }
        }
    }

    // Given: inner query returns fewer docs than maxResultsSize
    // When: RescoreRadialSearchQuery rescores
    // Then: iterator is used directly (no collectTopDocs), all valid results returned
    @SneakyThrows
    public void testRescore_whenCostBelowMaxResultsSize_thenUsesIteratorDirectly() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 1.0f, 0.0f, 0.0f },   // identical, similarity = 1.0
            { 0.9f, 0.1f, 0.0f },   // close, passes radius
            { 0.0f, 1.0f, 0.0f },   // orthogonal, fails radius — FALSE POSITIVE
        };
        final float radiusThreshold = 0.9f;
        // maxResultsSize = 10 > 3 docs, so direct iterator path is taken
        final int maxResultsSize = 10;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                Query innerQuery = new MatchAllDocsQuery();

                RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                    innerQuery,
                    FIELD_NAME,
                    queryVector,
                    radiusThreshold,
                    false,
                    maxResultsSize
                );

                TopDocs results = searcher.search(rescoreQuery, 10);

                // Only vectors within radius survive rescoring
                assertTrue("Expected some results within radius", results.scoreDocs.length > 0);
                assertTrue("False positive should be filtered", results.scoreDocs.length < vectors.length);
                for (ScoreDoc scoreDoc : results.scoreDocs) {
                    assertTrue("Score should be >= threshold, got " + scoreDoc.score, scoreDoc.score >= radiusThreshold);
                }
            }
        }
    }

    // Given: inner query returns more docs than the first-pass candidate limit
    // When: fewer results are requested than there are in-radius candidates
    // Then: firstPassK bounds the candidates rescored and the collector bounds the results returned,
    // which are the highest scoring ones
    @SneakyThrows
    public void testRescore_whenFewerResultsRequestedThanCandidates_thenCollectorBoundsResults() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        // 6 vectors with clearly different similarities to query.
        // Euclidean similarity = 1/(1+dist^2).
        // doc0: identical → score=1.0
        // doc1: very close → score≈0.99
        // doc2: close → score≈0.96
        // doc3-5: progressively farther → lower scores
        final float[][] vectors = {
            { 1.0f, 0.0f, 0.0f },    // doc0: score=1.0
            { 0.99f, 0.01f, 0.0f },  // doc1: very high score
            { 0.95f, 0.05f, 0.0f },  // doc2: high score
            { 0.7f, 0.3f, 0.0f },    // doc3: medium score
            { 0.5f, 0.5f, 0.0f },    // doc4: lower score
            { 0.3f, 0.7f, 0.0f },    // doc5: lowest score
        };
        // Radius loose enough that every vector clears the filter: minScore = 1/(1+1.0) = 0.5, and the
        // farthest doc (doc5) scores 1/(1+0.98) ~= 0.505.
        final float radiusThreshold = 1.0f;
        // Retain five first-pass candidates; ask the collector for only three results.
        final int firstPassK = 5;
        final int requestedResults = 3;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                // firstPassK bounds candidates per leaf, so a multi-segment index would trim nothing when
                // each leaf holds fewer than firstPassK docs. Force a single segment to make the per-leaf
                // bound observable as a per-query one.
                w.forceMerge(1);
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                Query innerQuery = new MatchAllDocsQuery();
                assertEquals("test assumes a single segment", 1, reader.leaves().size());

                RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                    innerQuery,
                    FIELD_NAME,
                    queryVector,
                    radiusThreshold,
                    false,
                    firstPassK
                );

                assertEquals(firstPassK, rescoreQuery.getFirstPassK());

                // Asking for everything yields the firstPassK candidates that pass the radius filter.
                assertEquals(firstPassK, searcher.search(rescoreQuery, vectors.length).scoreDocs.length);

                TopDocs results = searcher.search(rescoreQuery, requestedResults);

                assertEquals("Should return exactly the requested number of results", requestedResults, results.scoreDocs.length);

                // Results are the highest scoring of the rescored candidates, and every one clears the
                // radius. Which candidates survive the first-pass trim is not asserted: the inner
                // MatchAllDocsQuery scores every doc identically, so that selection is arbitrary.
                final float minScore = 1.0f / (1.0f + radiusThreshold);
                float previousScore = Float.MAX_VALUE;
                for (ScoreDoc scoreDoc : results.scoreDocs) {
                    assertTrue("score " + scoreDoc.score + " should clear the radius (" + minScore + ")", scoreDoc.score >= minScore);
                    assertTrue("results should be in descending score order", scoreDoc.score <= previousScore);
                    previousScore = scoreDoc.score;
                }
            }
        }
    }

    // Given: more than MAX_RESULTS_RADIAL_RESCORING (10k) vectors, all within radius
    // When: RescoreRadialSearchQuery rescores
    // Then: results are capped at MAX_RESULTS_RADIAL_RESCORING
    @SneakyThrows
    public void testRescore_whenResultsExceedMaxResultWindow_thenCappedAt10k() {
        // Given: index 10,050 vectors all very close to query (all within radius)
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final int totalDocs = 10_050;
        final float radiusThreshold = 0.01f; // Very loose — all docs should be within radius

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (int i = 0; i < totalDocs; i++) {
                    Document doc = new Document();
                    // All vectors identical to query — guaranteed similarity=1.0, always within any radius
                    doc.add(new KnnFloatVectorField(FIELD_NAME, queryVector.clone(), VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.forceMerge(1);
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                assertEquals("Expected single segment", 1, reader.leaves().size());
                {

                    IndexSearcher searcher = new IndexSearcher(reader);
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    // Search with a large collector size to not limit from the collector side
                    TopDocs results = searcher.search(rescoreQuery, totalDocs);

                    // Then: all 10,050 vectors pass the radius, but results are capped at 10,000
                    assertEquals("Results should be capped at MAX_RESULTS_RADIAL_RESCORING", 10_000, results.scoreDocs.length);
                }
            }
        }
    }

    // Given: more than 10k vectors, half within radius and half outside
    // When: RescoreRadialSearchQuery rescores
    // Then: only the valid half is returned (capped at 10k since valid count > 10k is impossible here,
    // but the invalid half is filtered out)
    @SneakyThrows
    public void testRescore_whenResultsExceedMaxResultWindow_andHalfAreValid_thenReturnsOnlyValid() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final int totalDocs = 10_050;
        final int validCount = totalDocs / 2; // ~5025 valid
        // Euclidean similarity = 1/(1+dist^2). For vectors close to query, score ≈ 1.0.
        // For orthogonal vectors, dist^2 ≈ 2, score ≈ 0.33.
        final float radiusThreshold = 0.9f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (int i = 0; i < totalDocs; i++) {
                    Document doc = new Document();
                    float[] vector;
                    if (i < validCount) {
                        // Valid: very close to query vector, will pass radius
                        vector = new float[] { 1.0f, i * 0.00001f, 0.0f };
                    } else {
                        // Invalid: orthogonal to query, will NOT pass radius
                        vector = new float[] { 0.0f, (float) Math.sin(i), (float) Math.cos(i) };
                    }
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, totalDocs);

                    // Then: only the valid half is returned, invalid half filtered out
                    assertEquals("Only valid vectors (first half) should survive rescoring", validCount, results.scoreDocs.length);
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        assertTrue(
                            "All returned docs should have score >= threshold, got " + scoreDoc.score,
                            scoreDoc.score >= radiusThreshold
                        );
                    }
                }
            }
        }
    }

    // Given: more than 10k vectors, ALL are outside the radius (all false positives)
    // When: RescoreRadialSearchQuery rescores
    // Then: all filtered out, empty result despite large first-pass set
    @SneakyThrows
    public void testRescore_whenResultsExceedMaxResultWindow_andAllAreFalsePositives_thenEmpty() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final int totalDocs = 10_050;
        // Very tight radius — nothing should survive since all vectors are far from query
        final float radiusThreshold = 0.9999f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (int i = 0; i < totalDocs; i++) {
                    Document doc = new Document();
                    // All vectors orthogonal/far from query — none will pass tight radius
                    float[] vector = { 0.0f, (float) Math.sin(i), (float) Math.cos(i) };
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, totalDocs);

                    // Then: all false positives filtered, empty result
                    assertEquals("All false positives should be filtered even with >10k candidates", 0, results.scoreDocs.length);
                }
            }
        }
    }

    // Given: vectors indexed with MAXIMUM_INNER_PRODUCT, inner query returns all (includes false positives)
    // When: RescoreRadialSearchQuery rescores with full precision
    // Then: only vectors with inner product score >= radius are kept
    @SneakyThrows
    public void testRescore_withMaxInnerProduct_filtersOutsideRadius() {
        final float[] queryVector = { 1.0f, 0.0f, 0.0f };
        final float[][] vectors = {
            { 3.0f, 0.0f, 0.0f },   // ip=3.0, Lucene score = 3.0+1 = 4.0
            { 1.0f, 0.0f, 0.0f },   // ip=1.0, Lucene score = 1.0+1 = 2.0
            { 0.1f, 0.0f, 0.0f },   // ip=0.1, Lucene score = 0.1+1 = 1.1
            { -1.0f, 0.0f, 0.0f },  // ip=-1.0, Lucene score = 1/(1-(-1)) = 0.5 — FALSE POSITIVE
            { -3.0f, 0.0f, 0.0f },  // ip=-3.0, Lucene score = 1/(1-(-3)) = 0.25 — FALSE POSITIVE
        };

        // Lucene MAX_INNER_PRODUCT score:
        // if ip >= 0: score = ip + 1
        // if ip < 0: score = 1 / (1 - ip)
        // Threshold 1.0 → only vectors with ip >= 0 pass (score >= 1.0)
        final float radiusThreshold = 1.0f;

        try (Directory directory = newDirectory()) {
            try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
                    w.addDocument(doc);
                }
                w.commit();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                {

                    IndexSearcher searcher = new IndexSearcher(reader);
                    Query innerQuery = new MatchAllDocsQuery();

                    RescoreRadialSearchQuery rescoreQuery = new RescoreRadialSearchQuery(
                        innerQuery,
                        FIELD_NAME,
                        queryVector,
                        radiusThreshold,
                        false,
                        MAX_RESULTS_RADIAL_RESCORING
                    );

                    TopDocs results = searcher.search(rescoreQuery, 10);

                    // Then: negative inner product vectors are filtered out
                    assertTrue("Expected some results", results.scoreDocs.length > 0);
                    assertTrue("Expected fewer than total (false positives removed)", results.scoreDocs.length < vectors.length);
                    for (ScoreDoc scoreDoc : results.scoreDocs) {
                        assertTrue("Score should be >= threshold, got " + scoreDoc.score, scoreDoc.score >= radiusThreshold);
                    }
                }
            }
        }
    }
}
