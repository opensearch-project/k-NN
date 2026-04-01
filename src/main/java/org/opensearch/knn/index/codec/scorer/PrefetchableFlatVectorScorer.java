/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
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
}
