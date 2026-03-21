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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;

/**
 * {@link FlatVectorsScorer} implementation for Asymmetric Distance Computation (ADC).
 * Scores float query vectors against quantized byte vectors stored in the index.
 * Supports {@link SpaceType#L2}, {@link SpaceType#INNER_PRODUCT}, and {@link SpaceType#COSINESIMIL}.
 */
public class ADCFlatVectorsScorer implements FlatVectorsScorer {
    private final SpaceType spaceType;

    /**
     * @param spaceType the space type used to determine the scoring formula
     */
    public ADCFlatVectorsScorer(SpaceType spaceType) {
        this.spaceType = spaceType;
    }

    /**
     * Not supported — ADC only operates on float query vectors against quantized byte vectors.
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues knnVectorValues,
        byte[] target
    ) {
        throw new UnsupportedOperationException("ADC does not support byte vector search");
    }

    /**
     * Returns a {@link RandomVectorScorer} that scores a float query vector against quantized byte vectors
     * using the ADC scoring formula for the configured {@link SpaceType}.
     *
     * @param vectorSimilarityFunction  ignored; space type is determined at construction time
     * @param knnVectorValues           must be an instance of {@link ByteVectorValues}
     * @param target                    the float query vector
     * @return a scorer computing ADC distance between {@code target} and each indexed byte vector
     * @throws IllegalArgumentException if {@code knnVectorValues} is not a {@link ByteVectorValues}
     */
    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction vectorSimilarityFunction,
        KnnVectorValues knnVectorValues,
        float[] target
    ) {
        if (!(knnVectorValues instanceof ByteVectorValues byteVectorValues)) {
            throw new IllegalArgumentException(
                "Expected " + ByteVectorValues.class.getSimpleName() + " for ADC scorer, got " + knnVectorValues.getClass().getSimpleName()
            );
        }

        return switch (spaceType) {
            case L2 -> new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                @Override
                public float score(int internalVectorId) throws IOException {
                    final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                    return SpaceType.L2.scoreTranslation(KNNScoringUtil.l2SquaredADC(target, quantizedByteVector));
                }
            };
            case INNER_PRODUCT, COSINESIMIL -> new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                @Override
                public float score(int internalVectorId) throws IOException {
                    final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                    return SpaceType.INNER_PRODUCT.scoreTranslation(-1 * KNNScoringUtil.innerProductADC(target, quantizedByteVector));
                }
            };
            default -> throw new IllegalArgumentException("Unsupported space type: " + spaceType);
        };
    }

    /**
     * Not supported — ADC does not support supplier-based scoring.
     */
    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) {
        throw new UnsupportedOperationException("ADC does not support RandomVectorScorerSupplier");
    }
}
