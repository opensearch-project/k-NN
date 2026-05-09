/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;

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
     * Constructs a new rescoring wrapper for radial search on a quantized index.
     *
     * @param innerQuery  the inner radial search query (must not be null)
     * @param field       the knn_vector field name (must not be null)
     * @param queryVector the query vector (must not be null)
     * @param radius      the radius threshold for the search
     */
    public RescoreRadialSearchQuery(final Query innerQuery, final String field, final float[] queryVector, float radius) {
        this.innerQuery = Objects.requireNonNull(innerQuery);
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        this.radius = radius;
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
            return new RescoreRadialSearchQuery(innerQuery.rewrite(indexSearcher), field, queryVector, radius);
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

        /**
         * @param query       the parent query (for Lucene's Weight contract)
         * @param innerWeight the inner weight from the quantized radial search query
         * @param boost       the score boost factor to apply to rescored results
         */
        RescoreWeight(Query query, Weight innerWeight, float boost) {
            super(query);
            this.innerWeight = innerWeight;
            this.boost = boost;
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
                    // Run radial search
                    final Scorer innerScorer = innerScorerSupplier.get(leadCost);
                    // TODO: Add rescoring with ExactSearcher using full-precision vectors.
                    // This will be dealt with in next PR.
                    return Objects.requireNonNullElseGet(innerScorer, KNNScorer::emptyScorer);
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
    }
}
