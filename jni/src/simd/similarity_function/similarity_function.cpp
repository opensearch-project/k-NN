/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#if defined(KNN_HAVE_AVX512_SPR)
    // Convert FP32 query vector to FP16 and do bulk
    // similarity with native AVX512-FP16 instructions.
    #include "avx512_spr_simd_similarity_function.cpp"
#elif defined(KNN_HAVE_AVX512)
    // Convert FP16 vectors to FP32 and do bulk similarity.
    #include "avx512_simd_similarity_function.cpp"
#elif defined(KNN_HAVE_ARM_FP16)
    #include "arm_neon_simd_similarity_function.cpp"
#else
    #include "default_simd_similarity_function.cpp"
#endif
