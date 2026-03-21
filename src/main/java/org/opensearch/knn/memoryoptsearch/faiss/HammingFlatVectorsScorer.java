/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;

/**
 * {@link FlatVectorsScorer} implementation for Hamming distance scoring.
 * Lucene does not provide a native Hamming distance scorer, so this class fills that gap.
 * Operates on byte vectors only.
 */
public class HammingFlatVectorsScorer implements FlatVectorsScorer {

    /**
     * Returns a {@link RandomVectorScorer} that computes Hamming distance between the byte query
     * vector and each indexed byte vector.
     *
     * @param vectorSimilarityFunction  ignored; Hamming distance is always used
     * @param knnVectorValues           must be an instance of {@link ByteVectorValues}
     * @param target                    the byte query vector
     * @return a scorer computing Hamming distance between {@code target} and each indexed byte vector
     * @throws IllegalArgumentException if {@code knnVectorValues} is not a {@link ByteVectorValues}
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues knnVectorValues,
        byte[] target
    ) {
        if (knnVectorValues instanceof ByteVectorValues byteVectorValues) {
            return new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                @Override
                public float score(int internalVectorId) throws IOException {
                    final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                    return KNNVectorSimilarityFunction.HAMMING.compare(target, quantizedByteVector);
                }
            };
        }

        throw new IllegalArgumentException(
            "Expected "
                + ByteVectorValues.class.getSimpleName()
                + " for hamming vector scorer, got "
                + knnVectorValues.getClass().getSimpleName()
        );
    }

    /**
     * Not supported — Hamming scoring does not support supplier-based scoring.
     */
    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues knnVectorValues
    ) {
        throw new UnsupportedOperationException();
    }

    /**
     * Not supported — Hamming distance operates on byte vectors only.
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues knnVectorValues,
        float[] target
    ) {
        throw new UnsupportedOperationException();
    }
}
