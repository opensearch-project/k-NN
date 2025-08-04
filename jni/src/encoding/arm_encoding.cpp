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
#if defined(__aarch64__) && defined(KNN_HAVE_ARM_FP16)
#include <arm_neon.h>
#endif

#include <cstdint>

#include "jni_util.h"
#include "encoding/arm_encoding.h"

jboolean knn_jni::encoding::isSIMDSupported() {
    return JNI_TRUE;
}

jboolean knn_jni::encoding::convertFP32ToFP16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    if (count <= 0) return JNI_TRUE;
    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        env->ReleasePrimitiveArrayCritical(fp16Array, dst_bytes, 0);
        env->ReleasePrimitiveArrayCritical(fp32Array, src_f32, JNI_ABORT);
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);


    int i = 0;
    // ARM NEON bulk 8-wide
    for (; i + 8 <= count; i += 8) {
        float32x4_t v0 = vld1q_f32(&src[i + 0]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 0]), h0);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 4]), h1);
    }
    // tail via NEON scalar broadcast
    for (; i < count; ++i) {
        float32x4_t sv = vdupq_n_f32(src[i]);
        float16x4_t hv = vcvt_f16_f32(sv);
        __fp16 lane = vget_lane_f16(hv, 0);
        dst[i] = *reinterpret_cast<const uint16_t*>(&lane);
    }

    jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
    jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    return JNI_TRUE;
}

