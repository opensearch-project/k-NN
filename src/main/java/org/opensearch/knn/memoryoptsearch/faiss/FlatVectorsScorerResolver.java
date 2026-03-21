/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;

import java.util.EnumMap;
import java.util.Map;

/**
 * Strategy interface for resolving a {@link FlatVectorsScorer} based on field metadata and similarity function.
 * Implementations are evaluated in order by {@link FlatVectorsScorerProvider}; the first one whose
 * {@link #canResolve} returns {@code true} wins. To add a new scorer, implement this interface as a
 * static inner class here and register it in {@link FlatVectorsScorerProvider}.
 */
public interface FlatVectorsScorerResolver {

    /**
     * Returns {@code true} if this resolver can handle the given field and similarity function.
     *
     * @param fieldInfo           the field metadata
     * @param similarityFunction  the similarity function for the query
     * @return {@code true} if this resolver applies
     */
    boolean canResolve(FieldInfo fieldInfo, KNNVectorSimilarityFunction similarityFunction);

    /**
     * Returns the {@link FlatVectorsScorer} for the given field.
     * Only called when {@link #canResolve} returns {@code true}.
     *
     * @param fieldInfo           the field metadata
     * @param similarityFunction  the similarity function for the query
     * @param delegateScorer      the default scorer, available for wrapping if needed
     * @return the resolved {@link FlatVectorsScorer}
     */
    FlatVectorsScorer resolve(FieldInfo fieldInfo, KNNVectorSimilarityFunction similarityFunction, FlatVectorsScorer delegateScorer);

    /**
     * Resolves to an {@link ADCFlatVectorsScorer} for fields using Asymmetric Distance Computation (ADC).
     * Selects the scorer based on the field's {@link SpaceType}.
     */
    class AdcScorerResolver implements FlatVectorsScorerResolver {
        private static final Map<SpaceType, FlatVectorsScorer> ADC_FLAT_SCORERS = new EnumMap<>(SpaceType.class);

        static {
            ADC_FLAT_SCORERS.put(SpaceType.L2, new ADCFlatVectorsScorer(SpaceType.L2));
            ADC_FLAT_SCORERS.put(SpaceType.COSINESIMIL, new ADCFlatVectorsScorer(SpaceType.COSINESIMIL));
            ADC_FLAT_SCORERS.put(SpaceType.INNER_PRODUCT, new ADCFlatVectorsScorer(SpaceType.INNER_PRODUCT));
        }

        @Override
        public boolean canResolve(FieldInfo fieldInfo, KNNVectorSimilarityFunction similarityFunction) {
            return FieldInfoExtractor.isAdc(fieldInfo);
        }

        @Override
        public FlatVectorsScorer resolve(
            FieldInfo fieldInfo,
            KNNVectorSimilarityFunction similarityFunction,
            FlatVectorsScorer delegateScorer
        ) {
            return ADC_FLAT_SCORERS.get(FieldInfoExtractor.getSpaceType(null, fieldInfo));
        }
    }

    /**
     * Resolves to a {@link HammingFlatVectorsScorer} when the similarity function is
     * {@link KNNVectorSimilarityFunction#HAMMING}. Used because Lucene does not provide a native Hamming scorer.
     */
    class HammingScorerResolver implements FlatVectorsScorerResolver {
        private static final FlatVectorsScorer HAMMING_FLAT_VECTORS_SCORER = new HammingFlatVectorsScorer();

        @Override
        public boolean canResolve(FieldInfo fieldInfo, KNNVectorSimilarityFunction similarityFunction) {
            return KNNVectorSimilarityFunction.HAMMING == similarityFunction;
        }

        @Override
        public FlatVectorsScorer resolve(
            FieldInfo fieldInfo,
            KNNVectorSimilarityFunction similarityFunction,
            FlatVectorsScorer delegateScorer
        ) {
            return HAMMING_FLAT_VECTORS_SCORER;
        }
    }

    /**
     * Resolves to a {@link Faiss104ScalarQuantizedVectorScorer} for fields using Faiss scalar quantization
     * with 1-bit quantization ({@link FaissSQEncoder.Bits#ONE}). Wraps the delegate scorer to be used as a fallback scorer
     * when SIMD acceleration is not applicable
     */
    class FaissSQScorerResolver implements FlatVectorsScorerResolver {

        @Override
        public boolean canResolve(FieldInfo fieldInfo, KNNVectorSimilarityFunction similarityFunction) {
            return FieldInfoExtractor.isSQField(fieldInfo)
                && FieldInfoExtractor.extractSQConfig(fieldInfo).getBits() == FaissSQEncoder.Bits.ONE.getValue();
        }

        @Override
        public FlatVectorsScorer resolve(
            FieldInfo fieldInfo,
            KNNVectorSimilarityFunction similarityFunction,
            FlatVectorsScorer delegateScorer
        ) {
            return new Faiss104ScalarQuantizedVectorScorer(delegateScorer);
        }
    }
}
