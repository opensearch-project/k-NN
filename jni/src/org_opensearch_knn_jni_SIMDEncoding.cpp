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

#include "org_opensearch_knn_jni_SIMDEncoding.h"

#include <jni.h>

#include "jni_util.h"
#include "encoding/encoding.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_SIMDENCODING_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_SIMDENCODING_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }
    jniUtil.Initialize(env);
    return KNN_SIMDENCODING_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SIMDENCODING_JNI_VERSION);
    jniUtil.Uninitialize(env);
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SIMDEncoding_convertFP32ToFP16(
    JNIEnv* env,
    jclass,
    jfloatArray fp32Array,
    jbyteArray fp16Array,
    jint count) {
    try {
        return knn_jni::encoding::convertFP32ToFP16(&jniUtil, env, fp32Array, fp16Array, count);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SIMDEncoding_isSIMDSupportedNative(
    JNIEnv*, jclass) {
    return knn_jni::encoding::isSIMDSupported();
}