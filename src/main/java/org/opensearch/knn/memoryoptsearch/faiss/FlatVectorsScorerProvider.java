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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

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

    public static class ADCFlatVectorsScorer implements FlatVectorsScorer {
        private final KNNVectorSimilarityFunction knnSimilarityFunction;
        private final SpaceType spaceType;

        public ADCFlatVectorsScorer(KNNVectorSimilarityFunction knnSimilarityFunction, SpaceType spaceType) {
            this.knnSimilarityFunction = knnSimilarityFunction;
            this.spaceType = spaceType;
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            byte[] target
        ) {
            throw new UnsupportedOperationException("ADC does not support byte vector search");
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            float[] target
        ) {
            if (knnVectorValues instanceof ByteVectorValues byteVectorValues) {
                if (spaceType == SpaceType.HAMMING) {
                    throw new IllegalArgumentException("hamming distance unsupported by ADC");
                } else if (spaceType == SpaceType.L2) {
                    return new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                        @Override
                        public float score(int internalVectorId) throws IOException {
                            final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                            return KNNScoringUtil.l2SquaredADC(target, quantizedByteVector);
                        }
                    };
                } else if (spaceType == SpaceType.COSINESIMIL) {
                    return new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                        @Override
                        public float score(int internalVectorId) throws IOException {
                            final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                            return KNNScoringUtil.innerProductADC(target, quantizedByteVector);
                        }
                    };
                } else if (spaceType == SpaceType.INNER_PRODUCT) {
                    return new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                        @Override
                        public float score(int internalVectorId) throws IOException {
                            final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                            return KNNScoringUtil.innerProductADC(target, quantizedByteVector);
                        }
                    };
                } else {
                    throw new IllegalArgumentException("Unsupported space type: " + spaceType);
                }
            }

            throw new IllegalArgumentException(
                "Expected " + ByteVectorValues.class.getSimpleName() + " for ADC scorer, got " + knnVectorValues.getClass().getSimpleName()
            );
        }

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues
        ) throws IOException {
            throw new UnsupportedOperationException("ADC does not support RandomVectorScorerSupplier");
        }
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
