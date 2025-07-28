/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
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
    private static final ADCFlatVectorsScorer ADC_FLAT_VECTORS_SCORER = new ADCFlatVectorsScorer();

    public static FlatVectorsScorer getFlatVectorsScorer(final KNNVectorSimilarityFunction similarityFunction) {
        if (similarityFunction == KNNVectorSimilarityFunction.HAMMING) {
            return HAMMING_VECTOR_SCORER;
        }

        return DELEGATE_VECTOR_SCORER;
    }

    public static ADCFlatVectorsScorer getAdcFlatVectorScorer(final KNNVectorSimilarityFunction similarityFunction) {
        return ADC_FLAT_VECTORS_SCORER;
    }

    public static class ADCFlatVectorsScorer implements FlatVectorsScorer {
        public RandomVectorScorer getRandomVectorScorerForAdc(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            float[] target,
            SpaceType spaceType
        ) {
            if (knnVectorValues instanceof ByteVectorValues byteVectorValues) {
                // TODO: do we need to worry about the score translation for each of the spaces in this case since it might not be handled
                // by rescoring in the hamming case?
                // also, we need an integration test with rescoring on and with rescoring off for ADC.
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
                "Cannot call the overridden function for adc."
                    + ByteVectorValues.class.getSimpleName()
                    + " for hamming vector scorer, got "
                    + knnVectorValues.getClass().getSimpleName()
            );
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            byte[] target
        ) {
            throw new IllegalArgumentException(
                "Cannot call the overridden function for adc."
                    + ByteVectorValues.class.getSimpleName()
                    + " for hamming vector scorer, got "
                    + knnVectorValues.getClass().getSimpleName()
            );
        }

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues
        ) throws IOException {
            return null;
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction vectorSimilarityFunction,
            KnnVectorValues knnVectorValues,
            float[] target
        ) {
            throw new IllegalArgumentException(
                "Cannot call the overridden function for adc."
                    + FloatVectorValues.class.getSimpleName()
                    + " for hamming vector scorer, got "
                    + knnVectorValues.getClass().getSimpleName()
            );
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
