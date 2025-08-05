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

#if defined(__aarch64__) && defined(KNN_HAVE_ARM_FP16)
  #pragma message("[KNN] Compiling ARM NEON FP16 encoding backend")
  #include "arm_encoding.cpp"
#elif defined(__x86_64__) && (defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_F16C))
  #pragma message("[KNN] Compiling x86 AVX encoding backend")
  #include "x86_encoding.cpp"
#else
  #pragma message("[KNN] Compiling default scalar encoding backend")
  #include "default_encoding.cpp"
#endif

