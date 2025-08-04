#
# SPDX-License-Identifier: Apache-2.0
# Copyright OpenSearch Contributors
#

include(CheckCXXSourceCompiles)

# Handle user overrides
if(NOT DEFINED AVX2_ENABLED)
    set(AVX2_ENABLED true)
endif()

if(NOT DEFINED AVX512_ENABLED)
    set(AVX512_ENABLED true)
endif()

if(NOT DEFINED AVX512_SPR_ENABLED)
    execute_process(
        COMMAND bash -c "lscpu | grep -q 'GenuineIntel' && lscpu | grep -i 'avx512_fp16' | grep -i 'avx512_bf16' | grep -i 'avx512_vpopcntdq'"
        OUTPUT_VARIABLE SPR_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT "${SPR_FLAGS}" STREQUAL "")
        set(AVX512_SPR_ENABLED true)
    else()
        set(AVX512_SPR_ENABLED false)
    endif()
endif()

# SIMD Detection
set(SIMD_OPT_LEVEL "")
set(SIMD_FLAGS "")

# SIMD feature flags default to OFF
set(KNN_HAVE_F16C OFF)
set(KNN_HAVE_AVX512 OFF)
set(KNN_HAVE_ARM_FP16 OFF)

# AVX512
if(AVX512_ENABLED)
    file(READ "/proc/cpuinfo" CPUINFO)
    string(FIND "${CPUINFO}" "avx512f" AVX512_CPU_FOUND)

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

    if(AVX512_CPU_FOUND GREATER -1 AND HAVE_AVX512_COMPILER)
        message(STATUS "[SIMD] AVX512F supported by CPU and compiler")
        set(SIMD_OPT_LEVEL "avx512")
        set(SIMD_FLAGS -mavx512f -mf16c)
        set(KNN_HAVE_AVX512 ON)
        add_definitions(-DKNN_HAVE_AVX512)
    else()
        message(STATUS "[SIMD] AVX512 skipped: CPU or compiler unsupported")
    endif()
endif()

# AVX2 + F16C
if(AVX2_ENABLED AND (SIMD_OPT_LEVEL STREQUAL ""))
    file(READ "/proc/cpuinfo" CPUINFO)
    string(FIND "${CPUINFO}" "avx2" AVX2_CPU_FOUND)
    string(FIND "${CPUINFO}" "f16c" F16C_CPU_FOUND)

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

    if(AVX2_CPU_FOUND GREATER -1 AND F16C_CPU_FOUND GREATER -1 AND HAVE_AVX2_COMPILER)
        message(STATUS "[SIMD] AVX2 + F16C supported by CPU and compiler")
        set(SIMD_OPT_LEVEL "avx2")
        set(SIMD_FLAGS -mavx2 -mf16c)
        set(KNN_HAVE_F16C ON)
        add_definitions(-DKNN_HAVE_F16C)
    else()
        message(STATUS "[SIMD] AVX2 skipped: CPU or compiler unsupported")
    endif()
endif()

# ARM NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND (SIMD_OPT_LEVEL STREQUAL ""))
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t v = vdupq_n_f32(1.0f);
            float16x4_t h = vcvt_f16_f32(v);
            (void)h;
            return 0;
        }" HAVE_NEON_COMPILER)

    if(HAVE_NEON_COMPILER)
        message(STATUS "[SIMD] ARM NEON FP16 supported by compiler (on ARM)")
        set(SIMD_OPT_LEVEL "neon")
        set(SIMD_FLAGS -march=armv8.4-a+fp16)
        set(KNN_HAVE_ARM_FP16 ON)
        add_definitions(-DKNN_HAVE_ARM_FP16)
    else()
        message(STATUS "[SIMD] NEON skipped: compiler unsupported")
    endif()
endif()

# Fallback
if(SIMD_OPT_LEVEL STREQUAL "")
    message(WARNING "[SIMD] No SIMD support detected or all SIMD options disabled. Falling back to Java encoding/decoding.")
    set(SIMD_OPT_LEVEL "generic")
    set(SIMD_FLAGS "")
endif()

# Always-used flags
set(FP16_SIMD_FLAGS "-O3" "-fPIC")