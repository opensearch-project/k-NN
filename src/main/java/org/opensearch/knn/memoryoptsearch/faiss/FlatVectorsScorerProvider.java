/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;

@UtilityClass
public class FlatVectorsScorerProvider {
    private static final FlatVectorsScorer DELEGATE_VECTOR_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
    private static final FlatVectorsScorer HAMMING_VECTOR_SCORER = new HammingFlatVectorsScorer();

    public static FlatVectorsScorer getFlatVectorsScorer(final KNNVectorSimilarityFunction similarityFunction) {
        if (similarityFunction == KNNVectorSimilarityFunction.HAMMING) {
            return HAMMING_VECTOR_SCORER;
        }

        return DELEGATE_VECTOR_SCORER;
    }

    private static class HammingFlatVectorsScorer implements FlatVectorsScorer {

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

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues
        ) {
            throw new UnsupportedOperationException();
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            float[] target
        ) {
            throw new UnsupportedOperationException();
        }
    }
}
