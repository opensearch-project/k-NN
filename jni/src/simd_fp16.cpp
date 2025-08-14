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

/* Selects the appropriate SIMD-optimized implementation of FP16
 * based on the target platform and available instruction sets at compile time.
 * - On ARM64 with FP16 support, includes arm_simd_fp16.cpp
 * - On x86_64 with AVX512 (SPR or general) or AVX2 + F16C, includes x86_simd_fp16.cpp
 * - Otherwise, falls back to the Java implementation in default_simd_fp16.cpp
 */
#if defined(__aarch64__) && defined(KNN_HAVE_ARM_FP16)
  #include "arm_simd_fp16.cpp"
#elif defined(__x86_64__) && (defined(KNN_HAVE_AVX512_SPR) || defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_AVX2_F16C))
  #include "x86_simd_fp16.cpp"
#else
  #include "default_simd_fp16.cpp"
#endif
