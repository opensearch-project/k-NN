/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Map;

@UtilityClass
public class FlatVectorsScorerProvider {
    private static final FlatVectorsScorer PREFETCHABLE_LUCENE99_SCORER = new PrefetchableFlatVectorScorer(
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
    );
    private static final FlatVectorsScorer HAMMING_VECTOR_SCORER = new PrefetchableFlatVectorScorer(new HammingFlatVectorsScorer());
    private static final Map<SpaceType, FlatVectorsScorer> ADC_FLAT_SCORERS = initializeAdcFlatScorers();

    /**
     * Returns Lucene's default flat vectors scorer wrapped with prefetching.
     * Reusable across any format or reader that needs a prefetch-enabled Lucene99 scorer.
     *
     * @return a prefetch-enabled {@link FlatVectorsScorer}
     */
    public static FlatVectorsScorer getLucene99FlatVectorsScorer() {
        return PREFETCHABLE_LUCENE99_SCORER;
    }

    private static Map<SpaceType, FlatVectorsScorer> initializeAdcFlatScorers() {
        Map<SpaceType, FlatVectorsScorer> scorers = new EnumMap<>(SpaceType.class);
        scorers.put(
            SpaceType.L2,
            new PrefetchableFlatVectorScorer(new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.EUCLIDEAN, SpaceType.L2))
        );
        scorers.put(
            SpaceType.COSINESIMIL,
            new PrefetchableFlatVectorScorer(new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.COSINE, SpaceType.COSINESIMIL))
        );
        scorers.put(
            SpaceType.INNER_PRODUCT,
            new PrefetchableFlatVectorScorer(
                new ADCFlatVectorsScorer(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, SpaceType.INNER_PRODUCT)
            )
        );
        return scorers;
    }

    /**
     * Returns the appropriate {@link FlatVectorsScorer} for the given field.
     * Selects an ADC, Hamming, or delegate scorer based on the field's quantization config and space type.
     *
     * @param fieldInfo           the field info containing space type and quantization attributes
     * @param similarityFunction  the similarity function used for scoring
     * @param delegateScorer      the default scorer to fall back to when no specialized scorer applies; must not be null
     * @return the resolved {@link FlatVectorsScorer}
     * @throws IllegalArgumentException if delegateScorer is null and no specialized scorer applies
     */
    public static FlatVectorsScorer getFlatVectorsScorer(
        final FieldInfo fieldInfo,
        final KNNVectorSimilarityFunction similarityFunction,
        final FlatVectorsScorer delegateScorer
    ) {
        // TODO: Refactor with a Resolver
        // Handle Special case of ADC first.
        if (FieldInfoExtractor.isAdc(fieldInfo)) {
            return ADC_FLAT_SCORERS.get(FieldInfoExtractor.getSpaceType(null, fieldInfo));
        } else if (KNNVectorSimilarityFunction.HAMMING == similarityFunction) {
            // Since Lucene doesn't provide hamming distance scorer, we return our own hamming distance scorer
            return HAMMING_VECTOR_SCORER;
        } else if (FieldInfoExtractor.isSQField(fieldInfo)
            && FieldInfoExtractor.extractSQConfig(fieldInfo).getBits() == FaissSQEncoder.Bits.ONE.getValue()) {
                return getKNN1040ScalarQuantizedVectorScorer(delegateScorer);
            } else if (delegateScorer != null) {
                // For all other cases, return the delegate scorer
                return delegateScorer;
            }
        throw new IllegalArgumentException("delegateScorer must not be null");
    }

    /**
     * Returns a {@link KNN1040ScalarQuantizedVectorScorer} that wraps the given delegate scorer.
     * This is the single entry point for creating the SIMD-accelerated SQ scorer, making it easy
     * to globally wrap or replace the scorer (e.g., with a prefetch-enabled variant).
     *
     * @param delegateScorer the fallback scorer used for non-quantized vectors
     * @return a SIMD-accelerated scalar quantized vector scorer
     */
    public static KNN1040ScalarQuantizedVectorScorer getKNN1040ScalarQuantizedVectorScorer(final FlatVectorsScorer delegateScorer) {
        return new KNN1040ScalarQuantizedVectorScorer(delegateScorer);
    }

    private static class ADCFlatVectorsScorer implements FlatVectorsScorer {
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

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues
        ) {
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
