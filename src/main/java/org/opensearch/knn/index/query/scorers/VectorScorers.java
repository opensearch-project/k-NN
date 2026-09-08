/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.common.Nullable;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

import static org.opensearch.knn.index.query.MemoryOptimizedSearchScoreConverter.convertInnerProductScoreToCosineScore;

/**
 * Static factory for creating {@link VectorScorer} instances from {@link KNNVectorValuesIterator.DocIdsIteratorValues}.
 *
 * <p>{@code VectorScorers} inspects the underlying iterator and vector values to select the appropriate
 * scoring strategy:
 * <ul>
 *   <li>{@link BinaryDocValues} → delegates to {@link KNNBinaryDocValuesScorer}</li>
 *   <li>{@link FloatVectorValues} → uses the provided {@link VectorScorerMode} (score or rescore),
 *       unless the configured space type disagrees with the similarity function recorded on the field</li>
 *   <li>{@link ByteVectorValues} with float target → ADC (Asymmetric Distance Computation) scoring</li>
 *   <li>{@link ByteVectorValues} with byte target → uses the provided {@link VectorScorerMode}, with the
 *       same space type override as the float case</li>
 * </ul>
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class VectorScorers {

    /**
     * Creates a {@link VectorScorer} for the given float query vector.
     *
     * @param docIdsIteratorValues wraps the {@link DocIdSetIterator} and {@link KnnVectorValues}
     *                             for the segment being scored
     * @param target    the float query vector
     * @param vectorScorerMode determines whether to use scoring or rescoring
     * @param spaceType the space type defining the similarity function
     * @param fieldInfo the field info for the vector field
     * @return a {@link VectorScorer} appropriate for the underlying vector storage format
     * @throws IOException if an I/O error occurs
     */
    public static VectorScorer createScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final float[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo
    ) throws IOException {
        return createScorer(docIdsIteratorValues, target, vectorScorerMode, spaceType, fieldInfo, null, null);
    }

    /**
     * Creates a {@link VectorScorer} for the given float query vector, wrapping with
     * {@link NestedBestChildVectorScorer} when nested search is required.
     *
     * @param docIdsIteratorValues wraps the {@link DocIdSetIterator} and {@link KnnVectorValues}
     *                             for the segment being scored
     * @param target    the float query vector
     * @param vectorScorerMode determines whether to use scoring or rescoring
     * @param spaceType the space type defining the similarity function
     * @param fieldInfo the field info for the vector field
     * @param filteredIdsIterator iterator over accepted child documents, or null if not nested
     * @param parentBitSet bit set identifying parent documents, or null if not nested
     * @return a {@link VectorScorer} appropriate for the underlying vector storage format
     * @throws IOException if an I/O error occurs
     */
    public static VectorScorer createScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final float[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo,
        @Nullable final DocIdSetIterator filteredIdsIterator,
        @Nullable final BitSet parentBitSet
    ) throws IOException {
        final VectorScorer scorer = getBaseScorer(docIdsIteratorValues, target, vectorScorerMode, spaceType, fieldInfo);
        return maybeWrapWithNestedScorer(scorer, filteredIdsIterator, parentBitSet);
    }

    /**
     * Creates a {@link VectorScorer} for the given byte query vector.
     *
     * @param docIdsIteratorValues wraps the {@link DocIdSetIterator} and {@link KnnVectorValues}
     *                             for the segment being scored
     * @param target    the byte query vector
     * @param vectorScorerMode determines whether to use scoring or rescoring
     * @param spaceType the space type defining the similarity function
     * @param fieldInfo the field info for the vector field
     * @return a {@link VectorScorer} appropriate for the underlying vector storage format
     * @throws IOException if an I/O error occurs
     */
    public static VectorScorer createScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final byte[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo
    ) throws IOException {
        return createScorer(docIdsIteratorValues, target, vectorScorerMode, spaceType, fieldInfo, null, null);
    }

    /**
     * Creates a {@link VectorScorer} for the given byte query vector, wrapping with
     * {@link NestedBestChildVectorScorer} when nested search is required.
     *
     * @param docIdsIteratorValues wraps the {@link DocIdSetIterator} and {@link KnnVectorValues}
     *                             for the segment being scored
     * @param target    the byte query vector
     * @param vectorScorerMode determines whether to use scoring or rescoring
     * @param spaceType the space type defining the similarity function
     * @param fieldInfo the field info for the vector field
     * @param acceptedChildrenIterator iterator over accepted child documents, or null if not nested
     * @param parentBitSet bit set identifying parent documents, or null if not nested
     * @return a {@link VectorScorer} appropriate for the underlying vector storage format
     * @throws IOException if an I/O error occurs
     */
    public static VectorScorer createScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final byte[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo,
        @Nullable final DocIdSetIterator acceptedChildrenIterator,
        @Nullable final BitSet parentBitSet
    ) throws IOException {
        final VectorScorer scorer = getBaseScorer(docIdsIteratorValues, target, vectorScorerMode, spaceType, fieldInfo);
        return maybeWrapWithNestedScorer(scorer, acceptedChildrenIterator, parentBitSet);
    }

    private static VectorScorer getBaseScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final float[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo
    ) throws IOException {
        final DocIdSetIterator docIdSetIterator = docIdsIteratorValues.getDocIdSetIterator();

        // ignore score mode, for BinaryDocValues since we do not support BinaryDocValues with quantization
        if (docIdSetIterator instanceof BinaryDocValues binaryDocValues) {
            return KNNBinaryDocValuesScorer.create(target, binaryDocValues, spaceType);
        }

        final KnnVectorValues knnVectorValues = docIdsIteratorValues.getKnnVectorValues();
        if (knnVectorValues instanceof FloatVectorValues floatVectorValues) {
            final VectorSimilarityFunction configuredFunction = resolveSimilarityFunction(spaceType);
            if (configuredFunction == null || configuredFunction == fieldInfo.getVectorSimilarityFunction()) {
                return vectorScorerMode.createScorer(floatVectorValues, target);
            }
            return createSimilarityOverrideScorer(floatVectorValues, target, configuredFunction);
        }
        if (knnVectorValues instanceof ByteVectorValues byteVectorValues && FieldInfoExtractor.isAdc(fieldInfo)) {
            return createADCScorer(fieldInfo, byteVectorValues, target, spaceType);
        }
        throw new IllegalArgumentException("Unsupported KnnVectorValues type: " + knnVectorValues.getClass());
    }

    private static VectorScorer getBaseScorer(
        final KNNVectorValuesIterator.DocIdsIteratorValues docIdsIteratorValues,
        final byte[] target,
        final VectorScorerMode vectorScorerMode,
        final SpaceType spaceType,
        final FieldInfo fieldInfo
    ) throws IOException {
        final DocIdSetIterator docIdSetIterator = docIdsIteratorValues.getDocIdSetIterator();

        // ignore score mode, for BinaryDocValues since we do not support BinaryDocValues with quantization
        if (docIdSetIterator instanceof BinaryDocValues binaryDocValues) {
            return KNNBinaryDocValuesScorer.create(target, binaryDocValues, spaceType);
        }

        final KnnVectorValues knnVectorValues = docIdsIteratorValues.getKnnVectorValues();
        if (knnVectorValues instanceof ByteVectorValues byteVectorValues) {
            if (spaceType == SpaceType.HAMMING) {
                return createHammingDistanceScorer(fieldInfo, byteVectorValues, target, spaceType);
            }
            final VectorSimilarityFunction configuredFunction = resolveSimilarityFunction(spaceType);
            if (configuredFunction == null || configuredFunction == fieldInfo.getVectorSimilarityFunction()) {
                return vectorScorerMode.createScorer(byteVectorValues, target);
            }
            return createSimilarityOverrideScorer(byteVectorValues, target, configuredFunction);
        }
        throw new IllegalArgumentException("Byte target requires ByteVectorValues but got " + knnVectorValues.getClass().getSimpleName());
    }

    private static VectorScorer maybeWrapWithNestedScorer(
        final VectorScorer scorer,
        @Nullable final DocIdSetIterator acceptedChildrenIterator,
        @Nullable final BitSet parentBitSet
    ) {
        if (parentBitSet == null) {
            return scorer;
        }
        return new NestedBestChildVectorScorer(acceptedChildrenIterator, parentBitSet, scorer);
    }

    /**
     * Resolves the Lucene {@link VectorSimilarityFunction} for the given space type.
     *
     * @param spaceType the configured space type
     * @return the matching similarity function, or {@code null} if the space type has no Lucene equivalent
     */
    // TODO: L1 and LINF have no Lucene similarity function, so they return null here and keep scoring with
    // the function recorded on the field. Correcting them needs a dedicated scorer, the way HAMMING has one.
    @Nullable
    private static VectorSimilarityFunction resolveSimilarityFunction(final SpaceType spaceType) {
        // L1, LINF and UNDEFINED report no similarity function, HAMMING throws when asked for one.
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = spaceType.getKnnVectorSimilarityFunction();
        if (knnVectorSimilarityFunction == null || knnVectorSimilarityFunction == KNNVectorSimilarityFunction.HAMMING) {
            return null;
        }
        return knnVectorSimilarityFunction.getVectorSimilarityFunction();
    }

    /**
     * Creates a {@link VectorScorer} that scores with the given similarity function instead of the one
     * recorded on the field.
     * <p>
     * The two differ when a field does not carry its space type into the recorded function.
     * {@code ModelFieldMapper} passes {@link SpaceType#DEFAULT} whatever the model was trained with, and
     * {@code FaissFieldStrategy} does the same for indices created in 2.17 through 2.19. Correcting the
     * recorded value is not an option for existing indices, since Lucene requires a field's vector
     * similarity to be identical across the segments of one index. Recording it correctly for newly
     * created indices, behind an index version gate, is left alone here.
     * <p>
     * This ignores {@link VectorScorerMode}, since neither a scorer nor a rescorer can be bound to a
     * different similarity function.
     * <p>
     * The Lucene99 scorer is what the per field formats reachable here delegate to anyway, so going through
     * {@code FlatVectorsScorerProvider#getFlatVectorsScorer} would select the same implementation.
     *
     * @param vectorValues the vector values for the segment
     * @param target the float query vector
     * @param similarityFunction the function to score with
     * @return a scorer over the given values
     * @throws IOException if an I/O error occurs
     */
    private static VectorScorer createSimilarityOverrideScorer(
        final FloatVectorValues vectorValues,
        final float[] target,
        final VectorSimilarityFunction similarityFunction
    ) throws IOException {
        return toVectorScorer(
            FlatVectorsScorerProvider.getLucene99FlatVectorsScorer().getRandomVectorScorer(similarityFunction, vectorValues, target),
            vectorValues
        );
    }

    /**
     * Byte counterpart of {@link #createSimilarityOverrideScorer(FloatVectorValues, float[], VectorSimilarityFunction)}.
     */
    private static VectorScorer createSimilarityOverrideScorer(
        final ByteVectorValues vectorValues,
        final byte[] target,
        final VectorSimilarityFunction similarityFunction
    ) throws IOException {
        return toVectorScorer(
            FlatVectorsScorerProvider.getLucene99FlatVectorsScorer().getRandomVectorScorer(similarityFunction, vectorValues, target),
            vectorValues
        );
    }

    /**
     * Adapts a {@link RandomVectorScorer} to the {@link VectorScorer} contract over the given values.
     */
    private static VectorScorer toVectorScorer(final RandomVectorScorer randomVectorScorer, final KnnVectorValues vectorValues) {
        return new VectorScorer() {
            final KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();

            @Override
            public float score() throws IOException {
                return randomVectorScorer.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
            }
        };
    }

    /**
     * Creates an ADC (Asymmetric Distance Computation) {@link VectorScorer} that scores a float query vector
     * against quantized byte document vectors.
     */
    // TODO: Remove once ByteVectorValues.scorer() is implemented to return the appropriate
    // VectorScorer based on ADC/quantization. At that point, VectorScorerMode.createScorer() will
    // handle this case and this method will no longer be needed.
    private static VectorScorer createADCScorer(
        final FieldInfo fieldInfo,
        final ByteVectorValues byteVectorValues,
        final float[] target,
        final SpaceType spaceType
    ) throws IOException {
        // We don't need to delegate since we know it is already ADC.
        // This will be removed once ADC Scorer is integrated into the reader.
        FlatVectorsScorer adcFlatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            spaceType.getKnnVectorSimilarityFunction(),
            null
        );
        // For COSINESIMIL, the ADCFlatVectorsScorer produces scores in INNER_PRODUCT format
        // (used by MemoryOptimizedKNNWeight which post-converts via convertToCosineScore).
        // In the exact search path there is no post-conversion, so we wrap the scorer to convert here.
        if (spaceType == SpaceType.COSINESIMIL) {
            adcFlatVectorsScorer = new CosineADCFlatVectorsScorer(adcFlatVectorsScorer);
        }
        final RandomVectorScorer randomVectorScorer = adcFlatVectorsScorer.getRandomVectorScorer(
            spaceType.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
            byteVectorValues,
            target
        );
        return new VectorScorer() {
            final KnnVectorValues.DocIndexIterator iterator = byteVectorValues.iterator();

            @Override
            public float score() throws IOException {
                return randomVectorScorer.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
            }
        };
    }

    /**
     * Wraps an ADC {@link FlatVectorsScorer} to convert INNER_PRODUCT-format scores to
     * COSINESIMIL-format. The ADCFlatVectorsScorer uses INNER_PRODUCT.scoreTranslation for
     * cosine, which the MemoryOptimized path post-converts. In the exact search path there
     * is no post-conversion, so this wrapper applies it at the scorer level.
     */
    // TODO: Move this cosine score conversion into ADCFlatVectorsScorer itself so that it directly
    // produces COSINESIMIL-format scores. This would eliminate the need for both this wrapper and
    // the post-conversion in MemoryOptimizedKNNWeight (convertToCosineScore), keeping the
    // conversion logic in a single place.
    private record CosineADCFlatVectorsScorer(FlatVectorsScorer delegate) implements FlatVectorsScorer {

        @Override
        public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues
        ) throws IOException {
            return delegate.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues,
            float[] target
        ) throws IOException {
            final RandomVectorScorer inner = delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
            return new RandomVectorScorer.AbstractRandomVectorScorer(vectorValues) {
                @Override
                public float score(int node) throws IOException {
                    return convertInnerProductScoreToCosineScore(inner.score(node));
                }
            };
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(
            VectorSimilarityFunction similarityFunction,
            KnnVectorValues vectorValues,
            byte[] target
        ) throws IOException {
            return delegate.getRandomVectorScorer(similarityFunction, vectorValues, target);
        }
    }

    /**
     * Creates a Hamming distance {@link VectorScorer} that scores a byte query vector
     * against binary byte document vectors using Hamming distance.
     *
     * @param fieldInfo         the field info for the vector field
     * @param byteVectorValues  the byte vector values from the segment
     * @param target            the byte query vector
     * @param spaceType         the space type defining the similarity function
     * @return a {@link VectorScorer} using Hamming distance scoring
     * @throws IOException if an I/O error occurs
     */
    // TODO: Remove once ByteVectorValues.scorer() is implemented to return the appropriate
    // VectorScorer based on the distance function. At that point, VectorScorerMode.createScorer()
    // will handle this case and this method will no longer be needed.
    private static VectorScorer createHammingDistanceScorer(
        final FieldInfo fieldInfo,
        final ByteVectorValues byteVectorValues,
        final byte[] target,
        final SpaceType spaceType
    ) throws IOException {
        final FlatVectorsScorer hammingFlatVectorsScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            spaceType.getKnnVectorSimilarityFunction(),
            null
        );
        // Hamming's KNNVectorSimilarityFunction does not map to a Lucene VectorSimilarityFunction,
        // but HammingFlatVectorsScorer ignores this parameter, so we pass EUCLIDEAN as a placeholder.
        final RandomVectorScorer randomVectorScorer = hammingFlatVectorsScorer.getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN,
            byteVectorValues,
            target
        );

        return new VectorScorer() {
            final KnnVectorValues.DocIndexIterator iterator = byteVectorValues.iterator();

            @Override
            public float score() throws IOException {
                return randomVectorScorer.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
            }
        };
    }
}
