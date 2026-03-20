/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;

/**
 * A Prefetchable {@link FlatVectorsScorer} that wraps another scorer and adds prefetching capabilities for doing
 * search. For all other cases it just delegates to the underlying scorer.
 */
@Log4j2
public class PrefetchableFlatVectorScorer implements FlatVectorsScorer {

    private final FlatVectorsScorer delegateScorer;

    /**
     * Constructs a new prefetchable scorer wrapper.
     *
     * @param delegateScorer the underlying scorer to delegate to
     */
    public PrefetchableFlatVectorScorer(final FlatVectorsScorer delegateScorer) {
        this.delegateScorer = delegateScorer;
    }

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        return delegateScorer.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        return new PrefetchableRandomVectorScorer(
            (RandomVectorScorer.AbstractRandomVectorScorer) delegateScorer.getRandomVectorScorer(similarityFunction, vectorValues, target)
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        return new PrefetchableRandomVectorScorer(
            (RandomVectorScorer.AbstractRandomVectorScorer) delegateScorer.getRandomVectorScorer(similarityFunction, vectorValues, target)
        );
    }

    @Override
    public String toString() {
        return "PrefetchableFlatVectorScorer()";
    }

    /**
     * A {@link RandomVectorScorer} that prefetches vector data before bulk scoring operations during search.
     *
     * <p>This scorer delegates all operations to an underlying scorer, but intercepts {@link
     * #bulkScore} to prefetch the required vectors before scoring.
     */
    @Log4j2
    static class PrefetchableRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {

        private final RandomVectorScorer.AbstractRandomVectorScorer delegate;

        /**
         * Constructs a new prefetchable random vector scorer.
         *
         * @param delegate the underlying scorer to delegate to
         */
        public PrefetchableRandomVectorScorer(final RandomVectorScorer.AbstractRandomVectorScorer delegate) {
            super(delegate.values());
            this.delegate = delegate;
        }

        @Override
        public float score(int node) throws IOException {
            return delegate.score(node);
        }

        /**
         * Scores multiple nodes with prefetching optimization.
         *
         * <p>Prefetches vector data before delegating to the underlying scorer for improved
         * performance.
         *
         * @param nodes    array of node ordinals to score
         * @param scores   output array for computed scores
         * @param numNodes number of nodes to score
         * @return the maximum score computed
         * @throws IOException if an I/O error occurs
         */
        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(values(), nodes, numNodes);
            return delegate.bulkScore(nodes, scores, numNodes);
        }

        @Override
        public int maxOrd() {
            return delegate.maxOrd();
        }

        @Override
        public int ordToDoc(int ord) {
            return delegate.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return delegate.getAcceptOrds(acceptDocs);
        }

        @Override
        public KnnVectorValues values() {
            return delegate.values();
        }
    }
}
