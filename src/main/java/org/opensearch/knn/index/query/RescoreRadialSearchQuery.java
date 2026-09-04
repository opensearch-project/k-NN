/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.Objects;

/**
 * A wrapper {@link Query} that adds full-precision rescoring to radial search on quantized indices
 * and exclude vectors if 'true distance' > radius.
 *
 * <p><b>Currently unreachable:</b> radial search on quantized indices is blocked by
 * {@link org.opensearch.knn.index.engine.ResolvedIndexSpec#supportsRadialSearch()} (returns
 * {@code false} for all quantized configurations), so no supported request path constructs
 * this query today. The implementation is retained, rather than deleted, for when
 * radial search on quantized indices is re-enabled behind a more robust scoring approach — see
 * <a href="https://github.com/opensearch-project/k-NN/issues/3452">#3452</a>.</p>
 *
 * <h2>Problem</h2>
 * <p>Radial search on quantized indices (e.g., 32x scalar quantization) computes similarity scores
 * using quantized vectors. These scores contain quantization error, which can produce <b>false
 * positives</b> — vectors whose quantized score falls within the user's radius but whose true
 * full-precision score does not.</p>
 *
 * <h2>Solution</h2>
 * <p>This query wraps the inner radial search query ({@link KNNQuery} for Faiss or
 * {@code FloatVectorSimilarityQuery} for Lucene) and adds a second-phase rescoring step.
 * The inner query performs the first-pass radial search on quantized vectors with the user's
 * radius. The wrapper then rescores the first-pass candidates using full-precision vectors
 * and filters out any results that fall outside the true radius.</p>
 *
 * @see RescoreKNNVectorQuery similar pattern for Lucene engine top-K rescoring
 * @see org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery similar pattern for Faiss engine top-K rescoring
 */
@Getter
@EqualsAndHashCode(callSuper = false)
public class RescoreRadialSearchQuery extends Query {
    private static ExactSearcher EXACT_SEARCHER_SINGLETON;

    /** The inner radial search query that operates on quantized vectors. */
    private final Query innerQuery;

    /** The name of the knn_vector field being searched. */
    private final String field;

    /** The original query vector provided by the user. */
    private final float[] queryVector;

    /**
     * The engine-specific radius threshold.
     * For Faiss, this is a raw distance converted via {@code KNNEngine.distanceToRadialThreshold()}.
     * For Lucene, this is a similarity value converted via {@code KNNEngine.scoreToRadialThreshold()}.
     */
    private final float radius;

    /**
     * Whether memory-optimized search is enabled for this field.
     * Determines how {@code radius} is interpreted during rescoring:
     * when true, radius is already a Lucene-normalized score;
     * when false, radius is a raw distance requiring conversion via {@code KNNEngine.score()}.
     */
    private final boolean memoryOptimizedSearchEnabled;

    /**
     * Maximum number of approximate candidates retained before full-precision rescoring.
     */
    private final int firstPassK;

    /**
     * Constructs a new rescoring wrapper for radial search on a quantized index.
     *
     * @param innerQuery                   the inner radial search query (must not be null)
     * @param field                        the knn_vector field name (must not be null)
     * @param queryVector                  the query vector (must not be null)
     * @param radius                       the radius threshold for the search
     * @param memoryOptimizedSearchEnabled whether memory-optimized search is enabled
     * @param firstPassK                   maximum number of approximate candidates to rescore
     */
    public RescoreRadialSearchQuery(
        final Query innerQuery,
        final String field,
        final float[] queryVector,
        float radius,
        final boolean memoryOptimizedSearchEnabled,
        final int firstPassK
    ) {
        this.innerQuery = Objects.requireNonNull(innerQuery);
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        this.radius = radius;
        this.memoryOptimizedSearchEnabled = memoryOptimizedSearchEnabled;
        this.firstPassK = firstPassK;
        Objects.requireNonNull(EXACT_SEARCHER_SINGLETON, "Exact searcher was not initialized.");
    }

    @VisibleForTesting
    public static void initialize(final ExactSearcher exactSearcher) {
        EXACT_SEARCHER_SINGLETON = exactSearcher;
    }

    /**
     * Creates a {@link RescoreWeight} that wraps the inner query's weight.
     * <p>The inner query is rewritten before weight creation to ensure any query optimizations
     * (e.g., constant folding) are applied.</p>
     *
     * @param searcher  the index searcher
     * @param scoreMode the score mode requested by the collector
     * @param boost     the boost factor to apply to rescored document scores
     * @return a weight that delegates scoring to the inner weight, with rescoring to be added
     * @throws IOException if an I/O error occurs during weight creation
     */
    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        final Weight innerWeight = searcher.createWeight(innerQuery, scoreMode, boost);
        return new RescoreWeight(this, innerWeight, boost);
    }

    @Override
    public Query rewrite(final IndexSearcher indexSearcher) throws IOException {
        final Query rewritten = innerQuery.rewrite(indexSearcher);
        if (rewritten != innerQuery) {
            return new RescoreRadialSearchQuery(rewritten, field, queryVector, radius, memoryOptimizedSearchEnabled, firstPassK);
        } else {
            return this;
        }
    }

    @Override
    public String toString(final String field) {
        return "RescoreRadialSearchQuery[field=" + this.field + ", radius=" + radius + ", innerQuery=" + innerQuery.toString(field) + "]";
    }

    /**
     * Propagates the visitor to the inner query as a MUST sub-clause, so that query analysis tools
     * (highlighting, field usage detection, profiling) can discover the inner radial search query
     * through this wrapper. Follows the same pattern as BoostQuery and ConstantScoreQuery.
     *
     * @param visitor
     */
    @Override
    public void visit(final QueryVisitor visitor) {
        innerQuery.visit(visitor.getSubVisitor(BooleanClause.Occur.MUST, this));
    }

    /**
     * Weight implementation that wraps the inner weight and provides per-leaf scorer suppliers.
     *
     * <p>The {@link #scorerSupplier(LeafReaderContext)} method returns a {@link ScorerSupplier}
     * whose {@code get()} method executes the full per-leaf pipeline:</p>
     * <ol>
     *   <li>Run the inner weight's scorer (quantized radial search on this leaf)</li>
     *   <li>Collect first-pass candidate doc IDs</li>
     *   <li>Rescore candidates with {@code ExactSearcher} using full-precision vectors</li>
     *   <li>Filter out docs whose true score falls outside the radius</li>
     *   <li>Return a {@link KNNScorer} over the final results</li>
     * </ol>
     *
     * <p>The {@code boost} factor is stored for use when constructing the final {@link KNNScorer},
     * which multiplies each document's score by the boost value.</p>
     */
    private static class RescoreWeight extends Weight {
        private final Weight innerWeight;
        private final float boost;
        private final String field;
        private final float[] queryVector;
        private final float radius;
        private final boolean memoryOptimizedSearchEnabled;
        private final int firstPassK;

        /**
         * @param query       the parent query (for Lucene's Weight contract)
         * @param innerWeight the inner weight from the quantized radial search query
         * @param boost       the score boost factor to apply to rescored results
         */
        RescoreWeight(Query query, Weight innerWeight, float boost) {
            super(query);
            this.innerWeight = innerWeight;
            this.boost = boost;
            RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
            this.field = rescoreQuery.field;
            this.queryVector = rescoreQuery.queryVector;
            this.radius = rescoreQuery.radius;
            this.memoryOptimizedSearchEnabled = rescoreQuery.memoryOptimizedSearchEnabled;
            this.firstPassK = rescoreQuery.firstPassK;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            return innerWeight.explain(context, doc);
        }

        /**
         * Returns a {@link ScorerSupplier} for the given leaf context.
         *
         * <p>Returns {@code null} if the inner weight has no scorer for this leaf (e.g., no
         * vectors indexed in this segment), following Lucene's convention.</p>
         *
         * @param context the leaf reader context for a single segment
         * @return a scorer supplier, or {@code null} if this segment has no candidates
         * @throws IOException if an I/O error occurs
         */
        @Override
        public ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
            final ScorerSupplier innerScorerSupplier = innerWeight.scorerSupplier(context);
            if (innerScorerSupplier == null) {
                return null;
            }
            return new ScorerSupplier() {
                long cost = -1;

                @Override
                public Scorer get(long leadCost) throws IOException {
                    // 1. Run inner scorer (quantized radial search on this leaf)
                    final Scorer innerScorer = innerScorerSupplier.get(leadCost);
                    if (innerScorer == null) {
                        return KNNScorer.emptyScorer();
                    }

                    // 2. Get matched docs from inner scorer
                    final DocIdSetIterator matchedDocs = innerScorer.iterator();
                    if (matchedDocs.cost() == 0) {
                        return KNNScorer.emptyScorer();
                    }

                    // 3. Retain at most the configured first-pass candidate count before exact rescoring.
                    // The inner scorer can exceed firstPassK: memory-optimized search widens the search
                    // to max(firstPassK, ef_search) and, when effectiveK == k, NativeEngineKnnVectorQuery
                    // merges the union of per-leaf results rather than trimming to k.
                    final DocIdSetIterator docsToRescore;
                    final long numDocsToRescore;
                    if (matchedDocs.cost() > firstPassK) {
                        final TopDocs topCandidates = collectTopDocs(innerScorer, firstPassK);
                        docsToRescore = new TopDocsDISI(topCandidates);
                        numDocsToRescore = topCandidates.scoreDocs.length;
                    } else {
                        docsToRescore = matchedDocs;
                        numDocsToRescore = matchedDocs.cost();
                    }

                    // 4. Build ExactSearcherContext — rescore with full-precision vectors
                    final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                        .matchedDocsIterator(docsToRescore)
                        .numberOfMatchedDocs(numDocsToRescore)
                        .useQuantizedVectorsForSearch(false)
                        .radius(radius)
                        .field(field)
                        .floatQueryVector(queryVector)
                        .isMemoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                        .build();

                    // 5. Return a lazy scorer over the candidates. exactSearchScorer hands back the
                    // BulkVectorScorer itself rather than draining it into TopDocs, so the
                    // full-precision vector reads happen only for the docs a conjunction actually
                    // advances to, instead of for every candidate up front.
                    final Scorer rescoreScorer = EXACT_SEARCHER_SINGLETON.exactSearchScorer(context, exactSearcherContext);
                    if (rescoreScorer == null) {
                        return KNNScorer.emptyScorer();
                    }
                    return boost == 1.0f ? rescoreScorer : new BoostedScorer(rescoreScorer, boost);
                }

                @Override
                public long cost() {
                    if (cost == -1) {
                        cost = innerScorerSupplier.cost();
                    }
                    return cost;
                }
            };
        }

        /**
         * Returns {@code true} because the rescore result is deterministic for the same
         * query parameters and segment state — safe to cache.
         */
        @Override
        public boolean isCacheable(final LeafReaderContext ctx) {
            return true;
        }

        /**
         * Collects the top candidateLimit documents by score from the scorer.
         */
        private TopDocs collectTopDocs(final Scorer scorer, final int candidateLimit) throws IOException {
            final TopKnnCollector collector = new TopKnnCollector(candidateLimit, Integer.MAX_VALUE);
            final DocIdSetIterator iterator = scorer.iterator();
            assert iterator.cost() > candidateLimit;
            int docId;
            while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
                collector.collect(docId, scorer.score());
            }
            return collector.topDocs();
        }
    }

    /**
     * Applies the query boost to a delegate scorer. Needed because the rescore pass now returns the
     * lazy {@code BulkVectorScorer} directly rather than a {@link KNNScorer}, which used to apply the
     * boost while replaying materialized results.
     */
    @VisibleForTesting
    static class BoostedScorer extends Scorer {
        private final Scorer delegate;
        private final float boost;

        BoostedScorer(final Scorer delegate, final float boost) {
            this.delegate = delegate;
            this.boost = boost;
        }

        @Override
        public int docID() {
            return delegate.docID();
        }

        @Override
        public DocIdSetIterator iterator() {
            return delegate.iterator();
        }

        @Override
        public float score() throws IOException {
            return delegate.score() * boost;
        }

        @Override
        public float getMaxScore(final int upTo) throws IOException {
            return delegate.getMaxScore(upTo) * boost;
        }

        @Override
        public int advanceShallow(final int target) throws IOException {
            return delegate.advanceShallow(target);
        }

        @Override
        public void setMinCompetitiveScore(final float minScore) throws IOException {
            delegate.setMinCompetitiveScore(boost == 0.0f ? 0.0f : minScore / boost);
        }
    }
}
