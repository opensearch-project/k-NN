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
