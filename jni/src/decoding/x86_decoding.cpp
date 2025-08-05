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
#if defined(__x86_64__) && (defined(KNN_HAVE_AVX512) || defined(KNN_HAVE_F16C))
#include <immintrin.h>
#endif

#include <cstdint>

#include "jni_util.h"
#include "decoding/decoding.h"

jboolean knn_jni::decoding::isSIMDSupported() {
    return JNI_TRUE;
}

jboolean knn_jni::decoding::convertFP16ToFP32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP32 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
        // Release the source FP16 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
    }};

    if ((reinterpret_cast<uintptr_t>(src_bytes + offset) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;  // release_arrays will still run its cleanup here
    }

    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    int i = 0;
#if defined(KNN_HAVE_AVX512)
    for (; i + 16 <= count; i += 16) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        __m512 v = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(&dst[i], v);
    }
#elif defined(KNN_HAVE_F16C)
    for (; i + 8 <= count; i += 8) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i]));
        __m256 v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(&dst[i], v);
    }
#endif
    for (; i < count; ++i) {
        __m128i h = _mm_cvtsi32_si128(src[i]);
        __m128 v = _mm_cvtph_ps(h);
        dst[i] = _mm_cvtss_f32(v);
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}