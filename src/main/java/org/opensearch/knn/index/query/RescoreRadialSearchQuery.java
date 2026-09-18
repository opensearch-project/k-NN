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
import org.apache.lucene.util.IOSupplier;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

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
 * <p>The {@code firstPassK} budget is applied across the shard, not per segment: see
 * {@link #createWeight}.</p>
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
     * <p>The first pass is <b>not</b> run here. Bounding the candidate set to the shard requires
     * every leaf's results at once, but a {@link Weight} may be built and then never scored — for
     * {@code explain}, {@code isCacheable}, or a conjunction clause Lucene never reaches. So the
     * shard-wide pass is deferred to the first {@link RescoreWeight#scorerSupplier} call and
     * memoized from there.</p>
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
        return new RescoreWeight(this, innerWeight, boost, () -> collectFirstPassCandidates(searcher, innerWeight));
    }

    /**
     * Runs the first pass on every leaf and reduces the union of per-leaf candidates to the globally
     * highest-scoring {@link #firstPassK}, so the budget bounds the shard rather than each segment:
     * a shard with {@code S} segments rescores {@code firstPassK} vectors instead of
     * {@code S * firstPassK}, and the survivors are the ones competitive across the whole shard.
     *
     * <p>{@link ResultUtil#reduceToTopK} trims by min-competitive-score rather than by a hard count,
     * so candidates tied at that score are all kept and the total can exceed {@code firstPassK}.</p>
     *
     * @return per-leaf survivors, indexed by {@link LeafReaderContext#ord}
     */
    private List<PerLeafResult> collectFirstPassCandidates(final IndexSearcher searcher, final Weight innerWeight) throws IOException {
        final List<Callable<PerLeafResult>> tasks = new ArrayList<>();
        for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
            tasks.add(() -> collectLeafCandidates(innerWeight, leaf));
        }
        final List<PerLeafResult> perLeafResults = searcher.getTaskExecutor().invokeAll(tasks);
        ResultUtil.reduceToTopK(perLeafResults, firstPassK);
        return perLeafResults;
    }

    /**
     * Runs the inner (quantized) first pass on a single leaf, retaining at most {@link #firstPassK}
     * candidates by score.
     */
    private PerLeafResult collectLeafCandidates(final Weight innerWeight, final LeafReaderContext leaf) throws IOException {
        final ScorerSupplier innerScorerSupplier = innerWeight.scorerSupplier(leaf);
        if (innerScorerSupplier == null) {
            return PerLeafResult.empty();
        }
        final Scorer innerScorer = innerScorerSupplier.get(Long.MAX_VALUE);
        if (innerScorer == null) {
            return PerLeafResult.empty();
        }
        final TopKnnCollector collector = new TopKnnCollector(firstPassK, Integer.MAX_VALUE);
        final DocIdSetIterator iterator = innerScorer.iterator();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            collector.collect(docId, innerScorer.score());
        }
        return new PerLeafResult(null, 0, collector.topDocs(), PerLeafResult.SearchMode.APPROXIMATE_SEARCH);
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
     * Weight implementation that rescores each leaf's share of the shard-wide first-pass candidates
     * collected in {@link #createWeight}, dropping docs whose full-precision score falls outside the
     * radius.
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

        /**
         * Runs the shard-wide first pass. Deferred so that a weight which is never scored does no
         * vector work, and memoized so that the pass runs once no matter how many leaves ask for it.
         */
        private final IOSupplier<List<PerLeafResult>> firstPass;
        private List<PerLeafResult> candidatesByLeaf;

        /**
         * @param query       the parent query (for Lucene's Weight contract)
         * @param innerWeight the inner weight from the quantized radial search query
         * @param boost       the score boost factor to apply to rescored results
         * @param firstPass   supplies the shard-wide first-pass survivors, indexed by leaf ordinal
         */
        RescoreWeight(Query query, Weight innerWeight, float boost, IOSupplier<List<PerLeafResult>> firstPass) {
            super(query);
            this.innerWeight = innerWeight;
            this.boost = boost;
            this.firstPass = firstPass;
            RescoreRadialSearchQuery rescoreQuery = (RescoreRadialSearchQuery) query;
            this.field = rescoreQuery.field;
            this.queryVector = rescoreQuery.queryVector;
            this.radius = rescoreQuery.radius;
            this.memoryOptimizedSearchEnabled = rescoreQuery.memoryOptimizedSearchEnabled;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            return innerWeight.explain(context, doc);
        }

        /**
         * Runs the first pass on first use and returns the memoized result. Synchronized because
         * concurrent segment search calls {@link #scorerSupplier} from several slice threads.
         */
        private synchronized List<PerLeafResult> candidatesByLeaf() throws IOException {
            if (candidatesByLeaf == null) {
                candidatesByLeaf = firstPass.get();
            }
            return candidatesByLeaf;
        }

        /**
         * Returns a {@link ScorerSupplier} that rescores this leaf's first-pass survivors, or
         * {@code null} if the leaf has none (e.g., no vectors indexed in this segment, or none of its
         * candidates survived the shard-wide trim), following Lucene's convention.
         *
         * @param context the leaf reader context for a single segment
         * @return a scorer supplier, or {@code null} if this segment has no candidates
         */
        @Override
        public ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
            final List<PerLeafResult> perLeafResults = candidatesByLeaf();
            // Indexed by leaf ordinal, which is valid because the first pass iterated the leaves of
            // this same searcher's reader. The bound guards against an unexpected reader hierarchy.
            if (context.ord >= perLeafResults.size()) {
                return null;
            }
            final TopDocs candidates = perLeafResults.get(context.ord).getResult();
            if (candidates.scoreDocs.length == 0) {
                return null;
            }
            return new ScorerSupplier() {
                @Override
                public Scorer get(long leadCost) throws IOException {
                    // Rescore with full-precision vectors, dropping anything outside the true radius.
                    final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                        .matchedDocsIterator(new TopDocsDISI(candidates))
                        .numberOfMatchedDocs(candidates.scoreDocs.length)
                        .useQuantizedVectorsForSearch(false)
                        .radius(radius)
                        .field(field)
                        .floatQueryVector(queryVector)
                        .isMemoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                        .build();

                    // Return a lazy scorer over the candidates. exactSearchScorer hands back the
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
                    // Upper bound: the radius filter drops some candidates during the rescore, but
                    // knowing how many would mean running the rescore here, defeating the lazy
                    // scorer. Lucene treats cost() as an estimate, and over-reporting only makes a
                    // conjunction order this clause later than strictly optimal.
                    return candidates.scoreDocs.length;
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
