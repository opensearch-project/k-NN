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

#ifndef KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_
#define KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __declspec(restrict)
#else
#define RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
/**
 * Generic wrapper for GCC/Clang's __builtin_assume_aligned.
 * This tells the compiler that 'ptr' is guaranteed to be aligned to 'align' bytes.
 */
#define BUILTIN_ASSUME_ALIGNED(ptr, align) \
    (typeof(ptr))__builtin_assume_aligned((ptr), (align))
#else

/**
 * Fallback for other compilers (e.g., MSVC or others without __builtin_assume_aligned).
 * Returns the original pointer, relying on explicit aligned intrinsics like _mm512_load_ps.
 */
#define BUILTIN_ASSUME_ALIGNED(ptr, align) (ptr)
#endif

#endif //KNNPLUGIN_JNI_INCLUDE_MEMORY_UTIL_H_
