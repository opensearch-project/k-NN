/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
// Ensures this header file is included only once during compilation.
// Prevents duplicate definition errors if included multiple times.

#if defined(__GNUC__) || defined(__clang__)
    // These macros are for GCC and Clang compilers.
    // __builtin_expect() gives the compiler a hint about which branch is more likely
    // to be taken. This helps the compiler generate more efficient branch prediction code.
    //
    // Example:
    //   if (LIKELY(x > 0))  → compiler assumes condition is usually true
    //   if (UNLIKELY(x < 0)) → compiler assumes condition is usually false
    //
    // The '!!(x)' ensures that 'x' is treated as a boolean (0 or 1).
    #define LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    // Fallback for any other compiler: do nothing special.
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

#if defined(_MSC_VER)
    // Microsoft Visual C++
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    // Fallback for other compilers
    #define FORCE_INLINE inline
#endif // FORCE_INLINE_H

// AT_HWCAP bit for the Armv8.2 dot-product instructions (asm/hwcap.h), with a fallback for libc headers that
// predate it. Shared by the SQ indexing and search kernels' runtime CPU detection.
#if defined(__aarch64__) && defined(__linux__)
    #include <sys/auxv.h>
    #if defined(HWCAP_ASIMDDP)
        #define KNN_SQ_HWCAP_ASIMDDP HWCAP_ASIMDDP
    #else
        #define KNN_SQ_HWCAP_ASIMDDP (1UL << 20)
    #endif
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define HOT_SPOT [[gnu::hot]]
#else
    #define HOT_SPOT
#endif
