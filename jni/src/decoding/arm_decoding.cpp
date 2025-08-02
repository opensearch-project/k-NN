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
#include "decoding/arm_decoding.h"

jboolean knn_jni::decoding::isSIMDSupported() {
    return JNI_TRUE;
}

jboolean knn_jni::decoding::convertFP16ToFP32(JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset) {
    if (count <= 0) return JNI_TRUE;
    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    if ((reinterpret_cast<uintptr_t>(src_bytes + offset) % alignof(uint16_t)) != 0) {
        env->ReleasePrimitiveArrayCritical(fp16Array, src_bytes, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(fp32Array, dst_f32, 0);
        return JNI_FALSE;
    }

    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    int i = 0;
    for (; i + 8 <= count; i += 8) {
        float16x4_t h0 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 0]));
        float16x4_t h1 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 4]));
        float32x4_t v0 = vcvt_f32_f16(h0);
        float32x4_t v1 = vcvt_f32_f16(h1);
        vst1q_f32(&dst[i + 0], v0);
        vst1q_f32(&dst[i + 4], v1);
    }
    for (; i < count; ++i) {
        __fp16 half = *reinterpret_cast<const __fp16*>(&src[i]);
        float32x4_t fv = vcvt_f32_f16(vdup_n_f16(half));
        dst[i] = vgetq_lane_f32(fv, 0);
    }

    jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
    jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
    return JNI_TRUE;
}