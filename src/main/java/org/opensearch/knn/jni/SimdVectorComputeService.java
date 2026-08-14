/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

/**
 * A service that computes vector similarity using native SIMD acceleration.
 * This service relies on a shared native library that implements optimized SIMD instructions to achieve faster performance during
 * similarity computations. The library must be properly loaded and available on the system before invoking any methods
 * that depend on native code.
 */
public class SimdVectorComputeService {
    static {
        KNNLibraryLoader.loadSimdLibrary();
    }

    /**
     * Similarity calculation type to passed down to native code.
     */
    public enum SimilarityFunctionType {
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.
        FP16_MAXIMUM_INNER_PRODUCT,
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.EUCLIDEAN.
        FP16_L2,
        SQ_IP,
        SQ_L2
    }

    public static native void saveSQSearchContext(
        byte[] quantizedQuery,
        float queryLowerInterval,
        float queryUpperInterval,
        float queryAdditionalCorrection,
        int queryQuantizedComponentSum,
        long[] addressAndSize,
        int nativeFunctionTypeOrd,
        int dimension,
        float centroidDp
    );

    /**
     * With vector ids, performing bulk SIMD similarity calculations and put the results into `scores`.
     *
     * @param internalVectorIds Vectors to load for similarity calculations.
     * @param scores            Results will be put into this array.
     * @param numVectors        The number of valid vector ids in `internalVectorIds`. Therefore, this will put exactly `numVectors` result
     *                          values into `scores`.
     */
    public native static float scoreSimilarityInBulk(int[] internalVectorIds, float[] scores, int numVectors);

    /**
     * Before vector search starts, it persists required information into a storage. Those persisted information will be used during search.
     * This must be called prior to each search.
     *
     * @param query                  Query vector
     * @param addressAndSize         An array describing vector chunks, where each pair of elements represents a chunk.
     *                               addressAndSize[i] is the starting memory address of the j-th chunk,
     *                               and addressAndSize[i + 1] is the size (in bytes) of that chunk where i = 2 * j.
     *                               Ex: addressAndSize[6] is the starting memory address of 3rd chunk, addressAndSize[7] is the size of
     *                               that chunk.
     * @param nativeFunctionTypeOrd  Similarity function type index.
     */
    public native static void saveSearchContext(float[] query, long[] addressAndSize, int nativeFunctionTypeOrd);

    /**
     * Perform similarity search on a single vector.
     *
     * @param internalVectorId Vector id
     * @return Similarity score.
     */
    public native static float scoreSimilarity(int internalVectorId);

    /**
     * Computes similarity between a query vector and one or more FP16-encoded vectors held
     * directly in a Java byte[] (as opposed to a memory-mapped native chunk). This allows non-mmap
     * scoring paths to still benefit from native SIMD FP16 distance computation when the CPU
     * supports it.
     *
     * @param fp16Vectors            FP16-encoded vector bytes, `numVectors` vectors packed contiguously
     *                               (2 * dimension bytes per vector, little-endian)
     * @param numVectors             Number of vectors packed in `fp16Vectors`
     * @param internalVectorIds      Positional ids of the packed vectors. Callers build this once and
     *                               reuse/grow it across calls to avoid reallocating it on every native call.
     * @param scores                 Output array to fill with `numVectors` similarity scores
     * @return The maximum of the computed scores.
     */
    public native static float scoreSimilarityInBulkFromFp16Bytes(
        byte[] fp16Vectors,
        int numVectors,
        int[] internalVectorIds,
        float[] scores
    );
}
