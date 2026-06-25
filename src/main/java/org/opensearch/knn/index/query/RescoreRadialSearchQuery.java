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
     * Maximum number of results to retain after rescoring.
     * Derived from the index-level {@code max_result_window} setting when available,
     * otherwise defaults to {@code MAX_RESULTS_RADIAL_RESCORING}.
     * All first-pass candidates are still scored, but only the top results up to this cap are kept.
     */
    private final int maxResultsSize;

    /**
     * Constructs a new rescoring wrapper for radial search on a quantized index.
     *
     * @param innerQuery                   the inner radial search query (must not be null)
     * @param field                        the knn_vector field name (must not be null)
     * @param queryVector                  the query vector (must not be null)
     * @param radius                       the radius threshold for the search
     * @param memoryOptimizedSearchEnabled whether memory-optimized search is enabled
     */
    public RescoreRadialSearchQuery(
        final Query innerQuery,
        final String field,
        final float[] queryVector,
        float radius,
        final boolean memoryOptimizedSearchEnabled,
        final int maxResultsSize
    ) {
        this.innerQuery = Objects.requireNonNull(innerQuery);
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        this.radius = radius;
        this.memoryOptimizedSearchEnabled = memoryOptimizedSearchEnabled;
        this.maxResultsSize = maxResultsSize;
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
            return new RescoreRadialSearchQuery(rewritten, field, queryVector, radius, memoryOptimizedSearchEnabled, maxResultsSize);
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
        private final int maxResultsSize;

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
            this.maxResultsSize = rescoreQuery.maxResultsSize;
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

                    // 3. If more candidates than maxResultsSize, pull only top-maxResultsSize
                    // from the inner scorer; otherwise use the iterator directly.
                    final DocIdSetIterator docsToRescore;
                    final long numDocsToRescore;
                    if (matchedDocs.cost() > maxResultsSize) {
                        final TopDocs topCandidates = collectTopDocs(innerScorer);
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
                        .maxResultWindow((int) Math.min(maxResultsSize, numDocsToRescore))
                        .radius(radius)
                        .field(field)
                        .floatQueryVector(queryVector)
                        .isMemoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                        .build();

                    // 5. Rescore — ExactSearcher handles radius → minScore conversion internally
                    final TopDocs rescored = EXACT_SEARCHER_SINGLETON.searchLeaf(context, exactSearcherContext);

                    // 6. Return scorer over rescored results
                    return new KNNScorer(rescored, boost);
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
         * Collects the top-maxResultsSize documents by score from the scorer.
         */
        private TopDocs collectTopDocs(final Scorer scorer) throws IOException {
            final TopKnnCollector collector = new TopKnnCollector(maxResultsSize, Integer.MAX_VALUE);
            final DocIdSetIterator iterator = scorer.iterator();
            assert (iterator.cost() > maxResultsSize);
            int docId;
            while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
                collector.collect(docId, scorer.score());
            }
            return collector.topDocs();
        }
    }
}
