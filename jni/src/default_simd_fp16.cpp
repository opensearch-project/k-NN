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
#include <iostream>

#include <cstdint>

#include "jni_util.h"
#include "simd_fp16.h"

// Returns JNI_FALSE to indicate that SIMD acceleration is not supported
// to fall back to the Java implementation.
jboolean knn_jni::simd::isSIMDSupported() {
    return JNI_FALSE;
}

// Stub implementation for FP32 to FP16 conversion in the absence of SIMD support.
// Always returns JNI_FALSE to signal that SIMD-based conversion is not available.
jboolean knn_jni::codec::fp16::encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv*, jfloatArray, jbyteArray, jint) {
    return JNI_FALSE;
}

// Stub implementation for FP16 to FP32 conversion in the absence of SIMD support.
// Always returns JNI_FALSE to signal that SIMD-based conversion is not available.
jboolean knn_jni::codec::fp16::decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv*, jbyteArray, jfloatArray, jint, jint) {
    return JNI_FALSE;
}