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
#include <immintrin.h>
#include <cstdint>

#include "jni_util.h"
#include "simd/fp16_codec/fp16_codec.h"

namespace knn_jni::simd::fp16_codec {

jboolean isSIMDSupported() {
    return JNI_TRUE;
}

jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    if (count <= 0) return JNI_TRUE;

    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    knn_jni::JNIReleaseElements release_arrays{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    size_t i = 0;

    // process 64 elements per iteration (8x unrolled).
    for (; i + 64 <= static_cast<size_t>(count); i += 64) {
        // One prefetch per 64-element iteration, distance chosen empirically
        // (fixed 64 floats ahead)
        if (i + 64 < static_cast<size_t>(count)) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // _mm256_loadu_ps: load 8 float32 values (unaligned) into a 256-bit
        // vector register (__m256).
        __m256 v0 = _mm256_loadu_ps(&src[i]);
        __m256 v1 = _mm256_loadu_ps(&src[i + 8]);
        __m256 v2 = _mm256_loadu_ps(&src[i + 16]);
        __m256 v3 = _mm256_loadu_ps(&src[i + 24]);
        __m256 v4 = _mm256_loadu_ps(&src[i + 32]);
        __m256 v5 = _mm256_loadu_ps(&src[i + 40]);
        __m256 v6 = _mm256_loadu_ps(&src[i + 48]);
        __m256 v7 = _mm256_loadu_ps(&src[i + 56]);
        // _mm256_cvtps_ph: convert 8 packed float32 lanes in a __m256 to
        // 8 packed float16 (IEEE half) values stored in a 128-bit integer
        // vector (__m128i). Rounding flags control FP32 -> FP16 behavior.
        __m128i h0 = _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h1 = _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h2 = _mm256_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h3 = _mm256_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h4 = _mm256_cvtps_ph(v4, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h5 = _mm256_cvtps_ph(v5, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h6 = _mm256_cvtps_ph(v6, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h7 = _mm256_cvtps_ph(v7, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // _mm_storeu_si128: store 128-bit integer vector to memory (unaligned).
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 8]), h1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 16]), h2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 24]), h3);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 32]), h4);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 40]), h5);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 48]), h6);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i + 56]), h7);
    }

    // F16C tail: process 8 elements
    for (; i + 8 <= static_cast<size_t>(count); i += 8) {
        // Tail: load 8 floats, convert to 8 half-precision values, store.
        __m256 v = _mm256_loadu_ps(&src[i]);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h);
    }

    // Scalar fallback for remaining elements
    for (; i < static_cast<size_t>(count); ++i) {
        // Scalar fallback: move single float into low lane of __m128,
        // convert to half stored in __m128i, then extract lower 32-bit
        // integer and truncate to 16 bits for storage.
        __m128 sv = _mm_set_ss(src[i]);
        __m128i hv = _mm_cvtps_ph(sv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(hv));
    }

    return JNI_TRUE;
}

}  // namespace knn_jni::simd::fp16_codec
