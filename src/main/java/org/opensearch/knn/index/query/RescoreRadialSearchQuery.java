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
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/** Rescores quantized radial-search candidates against full-precision vectors. */
@Getter
@EqualsAndHashCode(callSuper = false)
public class RescoreRadialSearchQuery extends AbstractRescoreQuery {
    private static ExactSearcher EXACT_SEARCHER_SINGLETON;

    private final Query innerQuery;
    private final String field;
    private final float[] queryVector;
    private final float radius;
    private final boolean memoryOptimizedSearchEnabled;
    private final int firstPassK;
    private final boolean shardLevelRescoringDisabled;

    public RescoreRadialSearchQuery(
        final Query innerQuery,
        final String field,
        final float[] queryVector,
        final float radius,
        final boolean memoryOptimizedSearchEnabled,
        final int firstPassK
    ) {
        this(innerQuery, field, queryVector, radius, memoryOptimizedSearchEnabled, firstPassK, false);
    }

    public RescoreRadialSearchQuery(
        final Query innerQuery,
        final String field,
        final float[] queryVector,
        final float radius,
        final boolean memoryOptimizedSearchEnabled,
        final int firstPassK,
        final boolean shardLevelRescoringDisabled
    ) {
        super(QueryUtils.getInstance());
        this.innerQuery = Objects.requireNonNull(innerQuery);
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        this.radius = radius;
        this.memoryOptimizedSearchEnabled = memoryOptimizedSearchEnabled;
        this.firstPassK = firstPassK;
        this.shardLevelRescoringDisabled = shardLevelRescoringDisabled;
        Objects.requireNonNull(EXACT_SEARCHER_SINGLETON, "Exact searcher was not initialized.");
    }

    @VisibleForTesting
    public static void initialize(final ExactSearcher exactSearcher) {
        EXACT_SEARCHER_SINGLETON = exactSearcher;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) {
        return new LazyRescoreWeight(this, searcher, scoreMode, boost);
    }

    private Weight createDelegateWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        final Weight innerWeight = searcher.createWeight(innerQuery, ScoreMode.TOP_SCORES, 1.0f);
        final List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<PerLeafResult> perLeafResults = collectFirstPassCandidates(searcher, innerWeight, leaves);

        perLeafResults = rescore(
            searcher,
            leaves,
            perLeafResults,
            firstPassK,
            shardLevelRescoringDisabled,
            () -> ExactSearcher.ExactSearcherContext.builder()
                .radius(radius)
                .field(field)
                .floatQueryVector(queryVector)
                .maxResultWindow(firstPassK)
                .isMemoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled),
            null,
            EXACT_SEARCHER_SINGLETON::searchLeaf
        );

        final TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        int resultCount = 0;
        for (int i = 0; i < perLeafResults.size(); i++) {
            final TopDocs leafTopDocs = perLeafResults.get(i).getResult();
            resultCount += leafTopDocs.scoreDocs.length;
            for (ScoreDoc scoreDoc : leafTopDocs.scoreDocs) {
                scoreDoc.doc += leaves.get(i).docBase;
            }
            topDocs[i] = leafTopDocs;
        }
        if (resultCount == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        final TopDocs merged = TopDocs.merge(resultCount, topDocs);
        return queryUtils.createDocAndScoreQuery(searcher.getIndexReader(), merged).createWeight(searcher, scoreMode, boost);
    }

    private List<PerLeafResult> collectFirstPassCandidates(
        final IndexSearcher searcher,
        final Weight innerWeight,
        final List<LeafReaderContext> leaves
    ) throws IOException {
        final List<Callable<PerLeafResult>> tasks = new ArrayList<>(leaves.size());
        for (final LeafReaderContext leaf : leaves) {
            tasks.add(() -> collectLeafCandidates(innerWeight, leaf));
        }
        return searcher.getTaskExecutor().invokeAll(tasks);
    }

    private PerLeafResult collectLeafCandidates(final Weight innerWeight, final LeafReaderContext leaf) throws IOException {
        final Scorer innerScorer = innerWeight.scorer(leaf);
        if (innerScorer == null) {
            return PerLeafResult.empty();
        }
        final TopDocs candidates = collectTopDocs(innerScorer, firstPassK);
        if (candidates.scoreDocs.length == 0) {
            return PerLeafResult.empty();
        }
        return new PerLeafResult(null, 0, candidates, PerLeafResult.SearchMode.APPROXIMATE_SEARCH);
    }

    private static TopDocs collectTopDocs(final Scorer scorer, final int candidateLimit) throws IOException {
        final TopKnnCollector collector = new TopKnnCollector(candidateLimit, Integer.MAX_VALUE);
        final DocIdSetIterator iterator = scorer.iterator();
        for (int docId = iterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = iterator.nextDoc()) {
            collector.collect(docId, scorer.score());
        }
        return collector.topDocs();
    }

    /** Defers the shard-wide candidate reduction and rescore until this weight is used. */
    private static final class LazyRescoreWeight extends Weight {
        private final RescoreRadialSearchQuery query;
        private final IndexSearcher searcher;
        private final ScoreMode scoreMode;
        private final float boost;
        private volatile Weight delegate;

        private LazyRescoreWeight(
            final RescoreRadialSearchQuery query,
            final IndexSearcher searcher,
            final ScoreMode scoreMode,
            final float boost
        ) {
            super(query);
            this.query = query;
            this.searcher = searcher;
            this.scoreMode = scoreMode;
            this.boost = boost;
        }

        @Override
        public Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
            return getDelegate().explain(context, doc);
        }

        @Override
        public ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
            return getDelegate().scorerSupplier(context);
        }

        @Override
        public int count(final LeafReaderContext context) throws IOException {
            return getDelegate().count(context);
        }

        @Override
        public boolean isCacheable(final LeafReaderContext context) {
            return false;
        }

        private Weight getDelegate() throws IOException {
            Weight current = delegate;
            if (current == null) {
                synchronized (this) {
                    current = delegate;
                    if (current == null) {
                        current = query.createDelegateWeight(searcher, scoreMode, boost);
                        delegate = current;
                    }
                }
            }
            return current;
        }
    }

    @Override
    public Query rewrite(final IndexSearcher indexSearcher) throws IOException {
        final Query rewritten = innerQuery.rewrite(indexSearcher);
        if (rewritten == innerQuery) {
            return this;
        }
        return new RescoreRadialSearchQuery(
            rewritten,
            field,
            queryVector,
            radius,
            memoryOptimizedSearchEnabled,
            firstPassK,
            shardLevelRescoringDisabled
        );
    }

    @Override
    public String toString(final String field) {
        return "RescoreRadialSearchQuery[field=" + this.field + ", radius=" + radius + ", innerQuery=" + innerQuery.toString(field) + "]";
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        innerQuery.visit(visitor.getSubVisitor(BooleanClause.Occur.MUST, this));
    }
}
