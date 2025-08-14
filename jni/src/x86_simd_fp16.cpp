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

#include <jni.h>
#if defined(__x86_64__) && (defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_AVX2_F16C))
#include <immintrin.h>
#endif

#include <cstdint>

#include "jni_util.h"
#include "simd_fp16.h"

// Returns JNI_TRUE to indicate that SIMD support is enabled at compile time for x86 architecture.
jboolean knn_jni::simd::isSIMDSupported() {
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP32 values to FP16.
 * using x86_64 vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Intel Intrinsics Guide:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */
jboolean knn_jni::codec::fp16::encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    // Obtain direct in-place access to the Java byte[] backing store for FP16 results.
    // GetPrimitiveArrayCritical typically pins the Java heap object and returns its real address
    // (no copy on most JVMs, since they support pinning).
    // If pinning isn’t possible (rare on modern OpenJDK), it will fall back to copying,
    // so always check for NULL.
    // While pinned, GC compaction is disabled, so keep this critical region very short
    // to avoid stalling other threads or garbage collection.
    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP16 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
        // Release the source FP32 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    // Ensure the FP16 destination buffer is 2-byte aligned.
    // The JVM will almost always give you an 8 or at least 4-byte-aligned buffer for any primitive array,
    // so the base pointer (dst_bytes or src_bytes) is naturally aligned to alignof(uint16_t)==2.
    // A misalignment here is therefore extremely rare, but this guard prevents undefined behavior or crashes.
    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        // release_arrays will cleanup the arrays automatically
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    size_t i = 0;
#if defined(KNN_HAVE_AVX512_SPR)
    for (; i + 32 <= count; i += 32) {
        // Load two __m512 (16 FP32 values each) to process 32 elements total
        __m512 v0 = _mm512_loadu_ps(&src[i]);
        __m512 v1 = _mm512_loadu_ps(&src[i + 16]);

        // Convert to two __m512h halves (each holds 16 FP16 values)
        // _mm512_cvtps_ph is only available on Intel Sapphire Rapids and newer CPUs with AVX512FP16.
        __m512h h0 = _mm512_cvtps_ph(v0);
        __m512h h1 = _mm512_cvtps_ph(v1);

        // Store each half (cast down to __m256i for correct size)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), _mm256_castph512_ph256(h0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i + 16]), _mm256_castph512_ph256(h1));
    }
#elif defined(KNN_HAVE_AVX512)
    for (; i + 16 <= count; i += 16) {
        // Load 16 float values (16 x 32 bits = 512 bits) into a __m512 register.
        // AVX512 registers are 512 bits wide, so they can hold 16 float32 values at once.
        __m512 v = _mm512_loadu_ps(&src[i]);
        // Convert the 16 FP32 values to FP16 and pack them into a 256-bit register (__m256i).
        // This uses round-to-nearest-even and disables floating-point exceptions.
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Store the 16 packed FP16 values (256 bits) to memory.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), h);
    }
#elif defined(KNN_HAVE_AVX2_F16C)
    for (; i + 8 <= count; i += 8) {
        // Load 8 float values (8 x 32 bits = 256 bits) into a __m256 register.
        // F16C uses 256-bit registers to convert 8 FP32 values to FP16 in parallel.
        __m256 v = _mm256_loadu_ps(&src[i]);
        // Convert 8 FP32 values to FP16 and pack them into a 128-bit register (__m128i).
        // This uses round-to-nearest-even and disables floating-point exceptions.
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Stores 8 packed FP16 values (128 bits) into memory at &dst[i] using unaligned access.
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h);
    }
#endif
    // Scalar fallback using F16C for remaining elements.
    // This path is taken if any elements remain after vectorized processing.
    // Converts one FP32 float at a time to FP16.
    for (; i < count; ++i) {
        // Load scalar float into the lowest part of __m128 register.
        __m128 sv = _mm_set_ss(src[i]);
        // Convert FP32 to FP16
        __m128i hv = _mm_cvtps_ph(sv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Extract the lowest 16 bits as uint16_t
        dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(hv));
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP16 values to FP32.
 * using x86_64 vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Intel Intrinsics Guide:
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */
jboolean knn_jni::codec::fp16::decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    // Pin the destination Java float[] and get raw access to its memory
    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    // Pin the source Java byte[] and get raw access to its memory
    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP32 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
        // Release the source FP16 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
    }};

    // Ensure that the starting address is aligned to 2 bytes (required for correct uint16_t interpretation)
    if ((reinterpret_cast<uintptr_t>(src_bytes + offset) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;  // release_arrays will still run its cleanup here
    }

    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    size_t i = 0;
#if defined(KNN_HAVE_AVX512_SPR)
    for (; i + 32 <= count; i += 32) {
        // Prefetch 128 FP16 elements (256 bytes) ahead into L1 cache.
        // Each AVX512_SPR iteration processes 32 FP16 values (64 bytes), so this prefetch is 4 iterations ahead.
        // This prefetches 4 full cache lines (assuming 64-byte cache line size)
        if (i + 128 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 128]), _MM_HINT_T0);
        }

        // Load and convert first 16 FP16 values
        __m256h h0 = _mm256_loadu_ph(&src[i]);
        __m512 v0 = _mm512_cvtph_ps(h0);

        // Load and convert next 16 FP16 values
        __m256h h1 = _mm256_loadu_ph(&src[i + 16]);
        __m512 v1 = _mm512_cvtph_ps(h1);

        // Store 32 FP32 values to memory
        _mm512_storeu_ps(&dst[i], v0);
        _mm512_storeu_ps(&dst[i + 16], v1);
    }
#elif defined(KNN_HAVE_AVX512)
    for (; i + 16 <= count; i += 16) {
        // Prefetch 64 FP16 values (128 bytes) ahead into L1 cache.
        // Each loop iteration processes 16 FP16 values (32 bytes), so this prefetch is 4 iterations ahead.
        // This gives the CPU time to load 2 cache lines before the data is accessed.
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // Load 16 FP16 values (stored as 16-bit integers) into a 256-bit AVX register
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        // Convert 16 FP16 values to 16 FP32 values using AVX512 hardware support
        __m512 v = _mm512_cvtph_ps(h);
        // Store the 16 converted FP32 values into destination array
        _mm512_storeu_ps(&dst[i], v);
    }
#elif defined(KNN_HAVE_AVX2_F16C)
    for (; i + 8 <= count; i += 8) {
        // Prefetch 64 FP16 values (128 bytes) ahead into L1 cache.
        // Each AVX2 iteration processes 8 FP16 values (16 bytes), so this is 8 iterations ahead.
        // This prefetches 2 cache lines of upcoming data, ideal for avoiding stalls during vectorized loads.
        // It’s tuned for stride-1 access patterns in large arrays.
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // Load 8 FP16 values (stored as 16-bit integers) into a 128-bit AVX register
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i]));
        // Convert 8 FP16 values to 8 FP32 values using AVX2 + F16C instructions
        __m256 v = _mm256_cvtph_ps(h);
         // Store the 8 converted FP32 values into destination array
        _mm256_storeu_ps(&dst[i], v);
    }
#endif
    // Scalar fallback using F16C for remaining elements.
    // This path is taken if any elements remain after vectorized processing.
    // Converts one FP16 float at a time to FP32.
    for (; i < count; ++i) {
        // Load a single FP16 value into the lower 16 bits of an XMM register
        __m128i h = _mm_cvtsi32_si128(src[i]);
        // Convert the FP16 to a single-precision float (__m128)
        __m128 v = _mm_cvtph_ps(h);
        // Extract the lowest 32-bit float from __m128 and store it
        dst[i] = _mm_cvtss_f32(v);
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}