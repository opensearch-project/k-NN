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
import java.util.EnumMap;
import java.util.Map;

@UtilityClass
public class FlatVectorsScorerProvider {
    private static final FlatVectorsScorer DELEGATE_VECTOR_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
    private static final FlatVectorsScorer HAMMING_VECTOR_SCORER = new HammingFlatVectorsScorer();
    private static final Map<SpaceType, FlatVectorsScorer> ADC_FLAT_SCORERS = initializeAdcFlatScorers();

    private static Map<SpaceType, FlatVectorsScorer> initializeAdcFlatScorers() {
        Map<SpaceType, FlatVectorsScorer> scorers = new EnumMap<>(SpaceType.class);
        scorers.put(SpaceType.L2, new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.EUCLIDEAN, SpaceType.L2));
        scorers.put(SpaceType.COSINESIMIL, new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.COSINE, SpaceType.COSINESIMIL));
        scorers.put(
            SpaceType.INNER_PRODUCT,
            new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT)
        );
        return scorers;
    }

    /**
     * Returns the FlatVectorsScorer based on the similarity function.
     * @param similarityFunction the vector similarity function to use
     * @return FlatVectorsScorer instance
     */
    public static FlatVectorsScorer getFlatVectorsScorer(final KNNVectorSimilarityFunction similarityFunction) {
        return getFlatVectorsScorer(similarityFunction, false, null);
    }

    /**
     * Returns the FlatVectorsScorer based on the similarity function, or if adc is enabled based on the SpaceType.
     * @param similarityFunction the vector similarity function to use
     * @param isAdc whether ADC (Asymmetric Distance Computation) is enabled
     * @param spaceType the space type for vector comparison
     * @return FlatVectorsScorer instance
     */
    public static FlatVectorsScorer getFlatVectorsScorer(
        final KNNVectorSimilarityFunction similarityFunction,
        final boolean isAdc,
        final SpaceType spaceType
    ) {
        if (isAdc) {
            // Note: we cannot leverage KNNVectorSimilarityFunction here as it is HAMMING for ADC, so we must use SpaceType.
            return ADC_FLAT_SCORERS.get(spaceType);
        }
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
            if (!(knnVectorValues instanceof ByteVectorValues byteVectorValues)) {
                throw new IllegalArgumentException(
                    "Expected "
                        + ByteVectorValues.class.getSimpleName()
                        + " for ADC scorer, got "
                        + knnVectorValues.getClass().getSimpleName()
                );
            }

            return switch (spaceType) {
                case HAMMING -> throw new IllegalArgumentException("hamming distance unsupported by ADC");
                case L2 -> new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                    @Override
                    public float score(int internalVectorId) throws IOException {
                        final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                        return SpaceType.L2.scoreTranslation(KNNScoringUtil.l2SquaredADC(target, quantizedByteVector));
                    }
                };
                case COSINESIMIL -> new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                    @Override
                    public float score(int internalVectorId) throws IOException {
                        final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                        return SpaceType.COSINESIMIL.scoreTranslation(1 - KNNScoringUtil.innerProductADC(target, quantizedByteVector));
                    }
                };
                case INNER_PRODUCT -> new RandomVectorScorer.AbstractRandomVectorScorer(knnVectorValues) {
                    @Override
                    public float score(int internalVectorId) throws IOException {
                        final byte[] quantizedByteVector = byteVectorValues.vectorValue(internalVectorId);
                        return SpaceType.INNER_PRODUCT.scoreTranslation(-1 * KNNScoringUtil.innerProductADC(target, quantizedByteVector));
                    }
                };
                default -> throw new IllegalArgumentException("Unsupported space type: " + spaceType);
            };
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
