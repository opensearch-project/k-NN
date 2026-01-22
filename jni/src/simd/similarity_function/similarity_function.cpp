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

#ifdef KNN_HAVE_AVX512
    #include "avx512_simd_similarity_function.cpp"
#elif KNN_HAVE_AVX512_SPR
    // Since we convert FP16 to FP32 then do bulk operation,
    // we're not really using SPR instruction set.
    // Therefore, both AVX512 and AVX512_SPR are sharing the same code piece.
    #include "avx512_simd_similarity_function.cpp"
#elif KNN_HAVE_ARM_FP16
    #include "arm_neon_simd_similarity_function.cpp"
#else
    #include "default_simd_similarity_function.cpp"
#endif
