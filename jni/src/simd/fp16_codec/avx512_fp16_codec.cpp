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
    knn_jni::JNIReleaseElements release_src{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));
    knn_jni::JNIReleaseElements release_dst{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
    }};

    // JVM byte arrays are always at least 8-byte aligned in practice, so this
    // should never trip; kept as a defensive check since every load/store
    // below is the unaligned variant (loadu/storeu) regardless of the result.
    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    size_t i = 0;

    // process 64 elements per iteration (4x unrolled)
    for (; i + 64 <= static_cast<size_t>(count); i += 64) {
        // One prefetch per 64-element iteration, distance chosen empirically
        // (fixed 64 floats ahead)
        if (i + 64 < static_cast<size_t>(count)) {
            // _mm_prefetch: hint to fetch the memory at the address into the
            // processor cache (T0 = temporal locality). Improves memory
            // throughput for the upcoming loads.
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // _mm512_loadu_ps: load 16 float32 values (unaligned) into a 512-bit
        // vector register (__m512).
        __m512 v0 = _mm512_loadu_ps(&src[i]);
        __m512 v1 = _mm512_loadu_ps(&src[i + 16]);
        __m512 v2 = _mm512_loadu_ps(&src[i + 32]);
        __m512 v3 = _mm512_loadu_ps(&src[i + 48]);
        // _mm512_cvtps_ph: convert 16 packed float32 lanes in a __m512 to
        // 16 packed float16 values packed into a 256-bit integer vector
        // The rounding mode flags control how FP32 -> FP16 rounding is performed.
        __m256i h0 = _mm512_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i h1 = _mm512_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i h2 = _mm512_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i h3 = _mm512_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // _mm256_storeu_si256: store a 256-bit vector to memory (unaligned).
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), h0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i + 16]), h1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i + 32]), h2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i + 48]), h3);
    }

    // Masked tail: up to four branch-free masked iterations for the remaining
    // 0-63 elements. The mask covers only in-bounds lanes, so the masked
    // load/store never reads or writes past `count`.
    for (; i < static_cast<size_t>(count); i += 16) {
        const size_t remaining = static_cast<size_t>(count) - i;
        const unsigned lanes = remaining < 16 ? static_cast<unsigned>(remaining) : 16u;
        const __mmask16 mask = static_cast<__mmask16>((1u << lanes) - 1u);
        // _mm512_maskz_loadu_ps: masked unaligned load of up to 16 float32
        // lanes from memory; lanes not selected by `mask` are zeroed.
        __m512 v = _mm512_maskz_loadu_ps(mask, &src[i]);
        // Convert the loaded float32 lanes to packed float16 values.
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // _mm256_mask_storeu_epi16: masked store of 16-bit lanes; only
        // elements with mask bits set are written to memory (unaligned).
        _mm256_mask_storeu_epi16(&dst[i], mask, h);
    }

    return JNI_TRUE;
}

jboolean decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jbyteArray fp16Array, jint offset, jfloatArray fp32Array, jint count) {
    if (count <= 0) return JNI_TRUE;

    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));
    knn_jni::JNIReleaseElements release_src{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
    }};

    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    knn_jni::JNIReleaseElements release_dst{[=]() {
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
    }};

    jbyte* src_bytes_off = src_bytes + offset;
    if ((reinterpret_cast<uintptr_t>(src_bytes_off) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;
    }

    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes_off);
    float* dst = reinterpret_cast<float*>(dst_f32);

    size_t i = 0;

    // process 64 elements per iteration (4x unrolled), mirrors encodeFp32ToFp16 above.
    for (; i + 64 <= static_cast<size_t>(count); i += 64) {
        if (i + 64 < static_cast<size_t>(count)) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        // _mm256_loadu_si256: load 16 packed float16 (as raw 16-bit lanes), unaligned.
        __m256i h0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        __m256i h1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i + 16]));
        __m256i h2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i + 32]));
        __m256i h3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i + 48]));
        // _mm512_cvtph_ps: convert 16 packed float16 lanes to 16 packed float32 lanes.
        __m512 v0 = _mm512_cvtph_ps(h0);
        __m512 v1 = _mm512_cvtph_ps(h1);
        __m512 v2 = _mm512_cvtph_ps(h2);
        __m512 v3 = _mm512_cvtph_ps(h3);
        // _mm512_storeu_ps: store 16 float32 values to memory (unaligned).
        _mm512_storeu_ps(&dst[i], v0);
        _mm512_storeu_ps(&dst[i + 16], v1);
        _mm512_storeu_ps(&dst[i + 32], v2);
        _mm512_storeu_ps(&dst[i + 48], v3);
    }

    // Masked tail: process the remaining 0-63 elements in up to four branch-free
    // masked iterations, mirrors the masked tail in encodeFp32ToFp16 above.
    for (; i < static_cast<size_t>(count); i += 16) {
        const size_t remaining = static_cast<size_t>(count) - i;
        const unsigned lanes = remaining < 16 ? static_cast<unsigned>(remaining) : 16u;
        const __mmask16 mask = static_cast<__mmask16>((1u << lanes) - 1u);
        // _mm256_maskz_loadu_epi16: masked unaligned load of up to 16 16-bit
        // lanes from memory; lanes not selected by `mask` are zeroed.
        __m256i h = _mm256_maskz_loadu_epi16(mask, &src[i]);
        __m512 v = _mm512_cvtph_ps(h);
        // _mm512_mask_storeu_ps: masked store of float32 lanes; only elements
        // with mask bits set are written to memory (unaligned).
        _mm512_mask_storeu_ps(&dst[i], mask, v);
    }

    return JNI_TRUE;
}

}  // namespace knn_jni::simd::fp16_codec
