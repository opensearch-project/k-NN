/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.NonNull;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.io.IOException;

/**
 * A {@link RandomVectorScorer} implementation that offloads vector similarity computation
 * to native SIMD-optimized code for maximum performance.
 * <p>
 * This class initializes a native search context based on the given query vector and
 * memory-mapped vector chunks, and delegates all similarity scoring operations to the
 * {@link SimdVectorComputeService}. The underlying native library is expected to
 * leverage SIMD instructions (e.g., AVX, AVX512, or NEON) to accelerate computations.
 */
public class NativeRandomVectorScorer implements RandomVectorScorer {

    // Backing {@link KnnVectorValues} used for document–vector association.
    @NonNull
    private final KnnVectorValues knnVectorValues;

    // Array of address–size pairs describing memory-mapped vector chunks.
    private long[] addressAndSize;

    // Maximum vector id available for scoring.
    private int maxOrd;

    // Index value of the native similarity function type.
    private int nativeFunctionTypeOrd;

    /**
     * Constructs a native-backed scorer for computing similarity between the given query
     * vector and a set of memory-mapped vectors.
     *
     * @param query                  the query vector represented as a {@code float[]} array
     * @param knnVectorValues        the {@link KnnVectorValues} providing document–vector mappings
     * @param mmapVectorValues       the {@link MMapVectorValues} describing native memory chunks
     * @param similarityFunctionType the similarity function type
     */
    public NativeRandomVectorScorer(
        final float[] query,
        final KnnVectorValues knnVectorValues,
        final MMapVectorValues mmapVectorValues,
        final SimdVectorComputeService.SimilarityFunctionType similarityFunctionType
    ) {
        this.knnVectorValues = knnVectorValues;
        this.addressAndSize = mmapVectorValues.getAddressAndSize();
        this.maxOrd = knnVectorValues.size();
        this.nativeFunctionTypeOrd = similarityFunctionType.ordinal();
        SimdVectorComputeService.saveSearchContext(query, addressAndSize, nativeFunctionTypeOrd);
    }

    /**
     * Computes similarity scores for multiple vectors in bulk using native SIMD code.
     *
     * @param internalVectorIds the array of internal vector IDs to score
     * @param scores            the output array to store computed similarity scores
     * @param numVectors        the number of vectors to process
     */
    @Override
    public void bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
        SimdVectorComputeService.scoreSimilarityInBulk(internalVectorIds, scores, numVectors);
    }

    /**
     * Computes the similarity score for a single vector using native SIMD code.
     *
     * @param internalVectorId the internal vector ID to score
     * @return the computed similarity score
     * @throws IOException if the native scoring operation fails
     */
    @Override
    public float score(final int internalVectorId) throws IOException {
        return SimdVectorComputeService.scoreSimilarity(internalVectorId);
    }

    /**
     * Returns the maximum vector id for scoring.
     *
     * @return the maximum vector id
     */
    @Override
    public int maxOrd() {
        return maxOrd;
    }

    /**
     * Maps an internal vector ordinal to its corresponding document ID.
     *
     * @param ord the internal vector id
     * @return the document ID associated with the given vector id
     */
    @Override
    public int ordToDoc(int ord) {
        return knnVectorValues.ordToDoc(ord);
    }

    /**
     * Returns a filtered {@link Bits} view representing accepted documents.
     *
     * @param acceptDocs the bit set of accepted documents
     * @return a {@link Bits} object describing acceptable vector ids for scoring
     */
    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
        return knnVectorValues.getAcceptOrds(acceptDocs);
    }
}
