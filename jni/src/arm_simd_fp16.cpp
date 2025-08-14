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
#include "simd_fp16.h"

// Returns JNI_TRUE to indicate that SIMD support is enabled at compile time for ARM architecture.
jboolean knn_jni::simd::isSIMDSupported() {
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP32 values to FP16
 * using ARM NEON vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Arm NEON Intrinsics Reference:
 * https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
 */
jboolean knn_jni::codec::fp16::encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    // Return early if there's nothing to convert
    if (count <= 0) return JNI_TRUE;

    // Pin the source Java float[] and get raw access to its memory
    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    // Pin the destination Java byte[] and get raw access to its memory
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));

    // When 'release_arrays' goes out of scope, its lambda will run to ensure that the
    // critical arrays are always released properly even if there's an error or early return
    knn_jni::JNIReleaseElements release_arrays{[=]() {
        // Release the destination FP16 array, mode 0 means changes are written back
        jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
        // Release the source FP32 array, JNI_ABORT means we don't write changes back (read-only)
        jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
    }};

    // Ensure that the starting address is aligned to 2 bytes (required for correct uint16_t interpretation)
    if ((reinterpret_cast<uintptr_t>(dst_bytes) % alignof(uint16_t)) != 0) {
        // release_arrays will cleanup the arrays automatically
        return JNI_FALSE;
    }

    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    int i = 0;
    // Each loop iteration handles 8 FP32 values by processing two 128-bit NEON registers,
    // each containing 4 FP32 values. These are converted into two sets of 4 FP16 values and stored.
    for (; i + 8 <= count; i += 8) {
        // Load 4 FP32 values (16 bytes) into 128-bit NEON register
        float32x4_t v0 = vld1q_f32(&src[i]);
        // Load next 4 FP32 values into another 128-bit NEON register
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        // Convert 4 FP32 values in v0 to 4 FP16 values using NEON half-precision conversion
        // Rounds to nearest even to match IEEE-754 behavior
        float16x4_t h0 = vcvt_f16_f32(v0);
        // Convert next 4 FP32 values
        float16x4_t h1 = vcvt_f16_f32(v1);
        // Store 4 FP16 values (8 bytes) from h0 into memory
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i]), h0);
        // Store next 4 FP16 values
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 4]), h1);
    }
    // Scalar conversion for remaining elements (count not divisible by 8)
    for (; i < count; ++i) {
        // Convert single FP32 to FP16 using scalar cast.
        // __fp16 is a native C++ type on ARM platforms supporting FP16.
        reinterpret_cast<__fp16*>(dst)[i] = static_cast<__fp16>(src[i]);
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}

/*
 * This function implements architecture-specific SIMD optimizations for converting FP16 values to FP32
 * using ARM NEON vector intrinsics. The conversion path is selected at compile time via preprocessor macros.
 * All of these intrinsics and instruction sets are documented in the official Arm NEON Intrinsics Reference:
 * https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
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

    int i = 0;
    // ARM NEON bulk conversion - processes 8 FP16 values per loop iteration in two 4-element blocks
    for (; i + 8 <= count; i += 8) {
        // Load 4 FP16 values (4 × 16 bits = 8 bytes) from memory into a 64-bit NEON register
        float16x4_t h0 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 0]));
        // Load next 4 FP16 values
        float16x4_t h1 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 4]));
        // Convert 4 FP16 values to 4 FP32 values using NEON vector conversion
        float32x4_t v0 = vcvt_f32_f16(h0);
        // Convert next 4 FP16 values
        float32x4_t v1 = vcvt_f32_f16(h1);
        // Store the first 4 FP32 values (4 × 32 bits = 16 bytes) to memory
        vst1q_f32(&dst[i + 0], v0);
        // Store the next 4 FP32 values
        vst1q_f32(&dst[i + 4], v1);
    }
    // Scalar conversion for remaining elements (count not divisible by 8)
    for (; i < count; ++i) {
        // Convert a single FP16 value to FP32 using scalar cast
        dst[i] = static_cast<float>(reinterpret_cast<const __fp16*>(src)[i]);
    }

    // Arrays are released automatically by the RAII release_arrays lambda
    return JNI_TRUE;
}
