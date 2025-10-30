/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;

/**
 * A service that computes vector similarity using native SIMD acceleration.
 * This service relies on a shared native library that implements optimized SIMD instructions to achieve faster performance during
 * similarity computations. The library must be properly loaded and available on the system before invoking any methods
 * that depend on native code.
 */
public class SimdVectorComputeService {
    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            // Even if the underlying system supports AVX512 and AVX2, users can override and disable it by setting
            // 'knn.faiss.avx2.disabled', 'knn.faiss.avx512.disabled', or 'knn.faiss.avx512_spr.disabled' to true in the opensearch.yml
            // configuration
            if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_COMPUTING_AVX512_SPR_JNI_LIBRARY_NAME);
            } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_COMPUTING_AVX512_JNI_LIBRARY_NAME);
            } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_COMPUTING_AVX2_JNI_LIBRARY_NAME);
            } else {
                System.loadLibrary(KNNConstants.DEFAULT_SIMD_COMPUTING_JNI_LIBRARY_NAME);
            }
            return null;
        });
    }

    /**
     * Similarity calculation type to passed down to native code.
     */
    public enum SimilarityFunctionType {
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.
        FP16_MAXIMUM_INNER_PRODUCT,
        // FP16 Maximum Inner Product. The result will be the same as we acquired from VectorSimilarityFunction.EUCLIDEAN.
        FP16_L2,
    }

    /**
     * With vector ids, performing bulk SIMD similarity calculations and put the results into `scores`.
     *
     * @param internalVectorIds Vectors to load for similarity calculations.
     * @param scores            Results will be put into this array.
     * @param numVectors        The number of valid vector ids in `internalVectorIds`. Therefore, this will put exactly `numVectors` result
     *                          values into `scores`.
     */
    public native static void scoreSimilarityInBulk(int[] internalVectorIds, float[] scores, int numVectors);

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
}
