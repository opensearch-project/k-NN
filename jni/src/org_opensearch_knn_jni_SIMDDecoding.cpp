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

#include "org_opensearch_knn_jni_SIMDDecoding.h"

#include <jni.h>

#include "jni_util.h"
#include "decoding/decoding.h"

static knn_jni::JNIUtil jniUtil;

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SIMDDecoding_convertFP16ToFP32(
    JNIEnv* env,
    jclass,
    jbyteArray fp16Array,
    jfloatArray fp32Array,
    jint count,
    jint offset) {
    try {
        return knn_jni::decoding::convertFP16ToFP32(&jniUtil, env, fp16Array, fp32Array, count, offset);
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_SIMDDecoding_isSIMDSupported(
    JNIEnv*, jclass) {
    return knn_jni::decoding::isSIMDSupported();
}
