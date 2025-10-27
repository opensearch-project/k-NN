#pragma once

#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#elif defined(_MSC_VER)
    // MSVC doesn't have __builtin_expect; just pass through
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif
