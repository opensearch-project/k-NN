#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXSourceCompiles)

# Allow user overrides
if(NOT DEFINED AVX2_ENABLED)
    set(AVX2_ENABLED true)   # set default value as true if the argument is not set
endif()

if(NOT DEFINED AVX512_ENABLED)
    set(AVX512_ENABLED true)   # set default value as true if the argument is not set
endif()

if(NOT DEFINED AVX512_SPR_ENABLED)
    # Check if the system is Intel(R) Sapphire Rapids or a newer-generation processor
    execute_process(COMMAND bash -c "lscpu | grep -q 'GenuineIntel' && lscpu | grep -i 'avx512_fp16' | grep -i 'avx512_bf16' | grep -i 'avx512_vpopcntdq'" OUTPUT_VARIABLE SPR_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT "${SPR_FLAGS}" STREQUAL "")
	      set(AVX512_SPR_ENABLED true)
    else()
	      set(AVX512_SPR_ENABLED false)
    endif()
endif()

# Default SIMD state
set(KNN_HAVE_AVX2_F16C OFF)
set(KNN_HAVE_AVX512 OFF)
set(KNN_HAVE_AVX512_SPR OFF)
set(KNN_HAVE_ARM_FP16 OFF)
set(SIMD_OPT_LEVEL "")
set(SIMD_FLAGS "")
set(SIMD_LIB_EXT "")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" OR (NOT AVX2_ENABLED AND NOT AVX512_ENABLED AND NOT AVX512_SPR_ENABLED))
    message(STATUS "[SIMD] Windows or SIMD explicitly disabled. Falling back to generic.")
    set(SIMD_OPT_LEVEL "generic")  # Keep optimization level as generic on Windows OS as it is not supported due to MINGW64 compiler issue.
    set(SIMD_FLAGS "")
    set(SIMD_LIB_EXT "")

elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64")
    set(CMAKE_REQUIRED_FLAGS "-march=armv8.4-a+fp16")
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t f = vdupq_n_f32(1.0f);
            float16x4_t h = vcvt_f16_f32(f);
            (void)h;
            return 0;
        }" HAVE_NEON_FP16)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_NEON_FP16)
        set(KNN_HAVE_ARM_FP16 ON)
        set(SIMD_OPT_LEVEL "generic") # On aarch64 avx2 is not supported.
        set(SIMD_FLAGS -march=armv8.4-a+fp16)
        set(SIMD_LIB_EXT "")
        add_definitions(-DKNN_HAVE_ARM_FP16)
        message(STATUS "[SIMD] ARM NEON with FP16 supported.")
    else()
        message(STATUS "[SIMD] ARM NEON FP16 instructions not supported by compiler. Falling back to generic.")
    endif()

elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND AVX512_SPR_ENABLED)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512fp16 -mf16c")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 v = _mm512_set1_ps(1.0f);
            __m256h h = _mm512_cvt_roundps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 w = _mm512_cvtph_ps(h);
            (void)w;
            return 0;
        }" HAVE_AVX512_SPR_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_AVX512_SPR_COMPILER)
        set(KNN_HAVE_AVX512_SPR ON)
        set(SIMD_OPT_LEVEL "avx512_spr")
        set(SIMD_FLAGS -mavx512f -mavx512fp16 -mf16c)
        set(SIMD_LIB_EXT "_avx512_spr")
        add_definitions(-DKNN_HAVE_AVX512_SPR)
        message(STATUS "[SIMD] AVX512_SPR supported by compiler.")
    else()
        message(FATAL_ERROR "[SIMD] AVX512_SPR was explicitly enabled, but compiler does not support it.")
    endif()

elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND AVX512_ENABLED)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mf16c")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m512 v = _mm512_setzero_ps();
            __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            (void)h;
            return 0;
        }" HAVE_AVX512_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_AVX512_COMPILER)
        set(KNN_HAVE_AVX512 ON)
        set(SIMD_OPT_LEVEL "avx512") # Keep optimization level as avx512 to improve performance on Linux. This is not present on mac systems, and presently not supported on Windows OS.
        set(SIMD_FLAGS -mavx512f -mf16c)
        set(SIMD_LIB_EXT "_avx512")
        add_definitions(-DKNN_HAVE_AVX512)
        message(STATUS "[SIMD] AVX512 + F16C supported by compiler.")
    else()
        message(FATAL_ERROR "[SIMD] AVX512 + FP16 was explicitly enabled, but compiler does not support it.")
    endif()

else()
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
            __m256 v = _mm256_setzero_ps();
            __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            (void)h;
            return 0;
        }" HAVE_AVX2_COMPILER)
    unset(CMAKE_REQUIRED_FLAGS)

    if(HAVE_AVX2_COMPILER)
        set(KNN_HAVE_AVX2_F16C ON)
        set(SIMD_OPT_LEVEL "avx2") # Keep optimization level as avx2 to improve performance on Linux and Mac.
        set(SIMD_FLAGS -mavx2 -mf16c)
        set(SIMD_LIB_EXT "_avx2")
        add_definitions(-DKNN_HAVE_AVX2_F16C)
        message(STATUS "[SIMD] AVX2 + F16C supported by compiler.")
    else()
        message(FATAL_ERROR "[SIMD] AVX2 + F16C was explicitly enabled, but compiler does not support it.")
    endif()
endif()

# Fallback if nothing matched
if(SIMD_OPT_LEVEL STREQUAL "")
    message(WARNING "[SIMD] No SIMD support detected or all SIMD options disabled. Falling back to Java encoding/decoding.")
    set(SIMD_OPT_LEVEL "generic")
    set(SIMD_FLAGS "")
    set(SIMD_LIB_EXT "")
endif()

# Always-used flags
set(FP16_SIMD_FLAGS "-O3" "-fPIC")