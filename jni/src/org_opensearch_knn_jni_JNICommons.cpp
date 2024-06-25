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

#include "org_opensearch_knn_jni_JNICommons.h"

#include <jni.h>
#include "commons.h"
#include "jni_util.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_JNICOMMONS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_JNICOMMONS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env);

    return KNN_JNICOMMONS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_JNICOMMONS_JNI_VERSION);
    jniUtil.Uninitialize(env);
}


JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_JNICommons_storeVectorData(JNIEnv * env, jclass cls,
jlong memoryAddressJ, jobjectArray dataJ, jlong initialCapacityJ)

{
    try {
        return knn_jni::commons::storeVectorData(&jniUtil, env, memoryAddressJ, dataJ, initialCapacityJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (long)memoryAddressJ;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_JNICommons_storeByteVectorData(JNIEnv * env, jclass cls,
jlong memoryAddressJ, jobjectArray dataJ, jlong initialCapacityJ)

{
    try {
        return knn_jni::commons::storeByteVectorData(&jniUtil, env, memoryAddressJ, dataJ, initialCapacityJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (long)memoryAddressJ;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_JNICommons_freeVectorData(JNIEnv * env, jclass cls,
                                                                            jlong memoryAddressJ)
{
    try {
        return knn_jni::commons::freeVectorData(memoryAddressJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}


JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_JNICommons_freeByteVectorData(JNIEnv * env, jclass cls,
                                                                            jlong memoryAddressJ)
{
    try {
        return knn_jni::commons::freeByteVectorData(memoryAddressJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}
