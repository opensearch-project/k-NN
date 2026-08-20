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
    #include "avx512_fp16_codec.cpp"
#elif KNN_HAVE_AVX512_SPR
    // Since we only do FP32<->FP16 conversion (not FP16 arithmetic),
    // AVX512 and AVX512_SPR share the same code piece.
    #include "avx512_fp16_codec.cpp"
#elif KNN_HAVE_AVX2_F16C
    #include "avx2_fp16_codec.cpp"
#elif KNN_HAVE_ARM_FP16
    #include "arm_neon_fp16_codec.cpp"
#else
    #include "default_fp16_codec.cpp"
#endif
