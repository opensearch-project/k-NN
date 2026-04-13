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
import org.opensearch.common.Nullable;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * Static factory for creating {@link VectorScorer} instances from {@link KNNVectorValuesIterator.DocIdsIteratorValues}.
 *
 * <p>{@code VectorScorers} inspects the underlying iterator and vector values to select the appropriate
 * scoring strategy:
 * <ul>
 *   <li>{@link BinaryDocValues} → delegates to {@link KNNBinaryDocValuesScorer}</li>
 *   <li>{@link FloatVectorValues} → uses the provided {@link VectorScorerMode} (score or rescore)</li>
 *   <li>{@link ByteVectorValues} with float target → ADC (Asymmetric Distance Computation) scoring</li>
 *   <li>{@link ByteVectorValues} with byte target → uses the provided {@link VectorScorerMode}</li>
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
            return vectorScorerMode.createScorer(floatVectorValues, target);
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
            return spaceType == SpaceType.HAMMING
                ? createHammingDistanceScorer(fieldInfo, byteVectorValues, target, spaceType)
                : vectorScorerMode.createScorer(byteVectorValues, target);
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
