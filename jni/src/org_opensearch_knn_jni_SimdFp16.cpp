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

#include "org_opensearch_knn_jni_SimdFp16.h"
#include "jni_util.h"
#include "simd/fp16_codec/fp16_codec.h"

static knn_jni::JNIUtil JNI_UTIL;

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SimdFp16_isSIMDSupportedNative
  (JNIEnv *env, jclass clazz) {
    return knn_jni::simd::fp16_codec::isSIMDSupported();
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SimdFp16_encodeFp32ToFp16
  (JNIEnv *env, jclass clazz, jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    try {
        return knn_jni::simd::fp16_codec::encodeFp32ToFp16(&JNI_UTIL, env, fp32Array, fp16Array, count);
    } catch (...) {
        JNI_UTIL.CatchCppExceptionAndThrowJava(env);
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SimdFp16_decodeFp16ToFp32
  (JNIEnv *env, jclass clazz, jbyteArray fp16Array, jint offset, jfloatArray fp32Array, jint count) {
    try {
        return knn_jni::simd::fp16_codec::decodeFp16ToFp32(&JNI_UTIL, env, fp16Array, offset, fp32Array, count);
    } catch (...) {
        JNI_UTIL.CatchCppExceptionAndThrowJava(env);
        return JNI_FALSE;
    }
}
