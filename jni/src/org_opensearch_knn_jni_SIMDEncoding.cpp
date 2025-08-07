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