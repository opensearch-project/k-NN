# SPDX-License-Identifier: Apache-2.0
# Copyright OpenSearch Contributors

include(CheckCXXSourceCompiles)

# Allow user overrides
if(NOT DEFINED AVX2_ENABLED)
    set(AVX2_ENABLED ON)
endif()

if(NOT DEFINED AVX512_ENABLED)
    set(AVX512_ENABLED ON)
endif()

if(NOT DEFINED AVX512_SPR_ENABLED)
    set(AVX512_SPR_ENABLED OFF)
endif()

# SIMD feature flags default to OFF
set(KNN_HAVE_F16C OFF)
set(KNN_HAVE_AVX512 OFF)
set(KNN_HAVE_ARM_FP16 OFF)

set(SIMD_OPT_LEVEL "")
set(SIMD_FLAGS "")

# AVX512 detection (compiler only)
if(AVX512_ENABLED)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mf16c")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 x = _mm512_setzero_ps();
            __m256i h = _mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            (void)h;
            return 0;
        }" HAVE_AVX512_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_AVX512_COMPILER)
        message(STATUS "[SIMD] AVX512F supported by compiler")
        set(SIMD_OPT_LEVEL "avx512")
        set(SIMD_FLAGS -mavx512f -mf16c)
        set(KNN_HAVE_AVX512 ON)
        add_definitions(-DKNN_HAVE_AVX512)
    else()
        message(STATUS "[SIMD] AVX512 skipped: compiler unsupported")
    endif()
endif()

# AVX2 + F16C detection
if(AVX2_ENABLED AND (SIMD_OPT_LEVEL STREQUAL ""))
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m128 x = _mm_setzero_ps();
            __m128i h = _mm_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            (void)h;
            return 0;
        }" HAVE_AVX2_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_AVX2_COMPILER)
        message(STATUS "[SIMD] AVX2 + F16C supported by compiler")
        set(SIMD_OPT_LEVEL "avx2")
        set(SIMD_FLAGS -mavx2 -mf16c)
        set(KNN_HAVE_F16C ON)
        add_definitions(-DKNN_HAVE_F16C)
    else()
        message(STATUS "[SIMD] AVX2 skipped: compiler unsupported")
    endif()
endif()

# ARM NEON + FP16 detection (used for AArch64/macOS)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND (SIMD_OPT_LEVEL STREQUAL ""))
    set(CMAKE_REQUIRED_FLAGS "-march=armv8.4-a+fp16")
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t v = vdupq_n_f32(1.0f);
            float16x4_t h = vcvt_f16_f32(v);
            (void)h;
            return 0;
        }" HAVE_NEON_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_NEON_COMPILER)
        message(STATUS "[SIMD] ARM NEON FP16 supported by compiler")
        set(SIMD_OPT_LEVEL "neon")
        set(SIMD_FLAGS -march=armv8.4-a+fp16)
        set(KNN_HAVE_ARM_FP16 ON)
        add_definitions(-DKNN_HAVE_ARM_FP16)
    else()
        message(STATUS "[SIMD] NEON skipped: compiler unsupported")
    endif()
endif()

# Fallback if no SIMD option was enabled
if(SIMD_OPT_LEVEL STREQUAL "")
    message(WARNING "[SIMD] No SIMD support detected or all SIMD options disabled. Falling back to Java encoding/decoding.")
    set(SIMD_OPT_LEVEL "generic")
    set(SIMD_FLAGS "")
endif()

# Always-used flags
set(FP16_SIMD_FLAGS "-O3" "-fPIC")