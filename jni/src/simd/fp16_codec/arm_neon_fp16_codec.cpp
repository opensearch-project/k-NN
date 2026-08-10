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
#include <arm_neon.h>
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

    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    size_t i = 0;

    // process 16 elements per iteration (2x unrolled)
    // Combines each pair of 4-lane conversion results into a single 8-lane register
    // so the loop does 2 stores of 8 lanes each.
    for (; i + 16 <= static_cast<size_t>(count); i += 16) {
        // __builtin_prefetch: software prefetch hint to bring future memory
        // into cache. First arg is the address, 0 = read, 3 = high locality.
        if (i + 128 < static_cast<size_t>(count)) {
            __builtin_prefetch(&src[i + 128], 0, 3);
        }
        // vld1q_f32: load 4 float32 values into a NEON 128-bit register.
        float32x4_t v0 = vld1q_f32(&src[i]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float32x4_t v2 = vld1q_f32(&src[i + 8]);
        float32x4_t v3 = vld1q_f32(&src[i + 12]);
        // vcvt_f16_f32: convert 4 float32 lanes to 4 float16 lanes.
        // vcombine_f16: combine two 4-lane float16 vectors into one 8-lane
        // float16 vector for more efficient storing.
        float16x8_t h01 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
        float16x8_t h23 = vcombine_f16(vcvt_f16_f32(v2), vcvt_f16_f32(v3));
        // vst1q_f16: store 8 float16 lanes to memory (aligned/unaligned
        // depending on pointer). Uses __fp16 pointer cast for storage.
        vst1q_f16(reinterpret_cast<__fp16*>(&dst[i]), h01);
        vst1q_f16(reinterpret_cast<__fp16*>(&dst[i + 8]), h23);
    }

    // NEON tail: process 8 elements (at most once, since the 16-lane loop above
    // already consumed every multiple of 16)
    if (i + 8 <= static_cast<size_t>(count)) {
        // Tail: load two 4-lane vectors, convert to float16 and store 8 lanes.
        float32x4_t v0 = vld1q_f32(&src[i]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float16x8_t h = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
        vst1q_f16(reinterpret_cast<__fp16*>(&dst[i]), h);
        i += 8;
    }

    // NEON tail: process 4 elements
    if (i + 4 <= static_cast<size_t>(count)) {
        // 4-lane tail: load 4 floats, convert to 4 float16 lanes, store.
        float32x4_t v0 = vld1q_f32(&src[i]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        // vst1_f16: store 4 float16 lanes to memory.
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i]), h0);
        i += 4;
    }

    // Scalar fallback for remaining elements
    for (; i < static_cast<size_t>(count); ++i) {
        // Scalar fallback: convert single float to __fp16 and write to dst.
        reinterpret_cast<__fp16*>(dst)[i] = static_cast<__fp16>(src[i]);
    }

    return JNI_TRUE;
}

}  // namespace knn_jni::simd::fp16_codec
