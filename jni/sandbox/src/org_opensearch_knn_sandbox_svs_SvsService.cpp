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

#include "org_opensearch_knn_sandbox_svs_SvsService.h"

#include <jni.h>

#include "jni_util.h"
#include "svs_wrapper.h"
#include "faiss_stream_support.h"
#include "faiss/impl/FaissException.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_SVS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_SVS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env, vm);

    return KNN_SVS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SVS_JNI_VERSION);
    if (faiss::InterruptCallback::instance.get() != nullptr) {
        faiss::InterruptCallback::instance.get()->clear_instance();
    }
    jniUtil.Uninitialize(env);
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_isLvqLeanvecEnabled(JNIEnv * env, jclass cls)
{
    try {
        return knn_jni::svs_wrapper::IsLvqLeanvecEnabled() ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return JNI_FALSE;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_initIndex(JNIEnv * env, jclass cls,
                                                                                 jlong numDocs, jint dimJ,
                                                                                 jobject parametersJ)
{
    try {
        return knn_jni::svs_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_insertToIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                    jlong vectorsAddressJ, jint dimJ,
                                                                                    jlong indexAddress, jint threadCount)
{
    try {
        knn_jni::svs_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchIndexBuildAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_writeIndex(JNIEnv * env,
                                                                                 jclass cls,
                                                                                 jlong indexAddress,
                                                                                 jobject output)
{
    try {
        knn_jni::svs_wrapper::WriteIndex(&jniUtil, env, output, indexAddress);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_loadIndexWithStream(JNIEnv * env,
                                                                                           jclass cls,
                                                                                           jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `readStream` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading the vector index.
        return knn_jni::svs_wrapper::LoadIndexWithStream(&faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return (jlong)0;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_queryIndex(JNIEnv * env, jclass cls,
                                                                                         jlong indexPointerJ,
                                                                                         jfloatArray queryVectorJ,
                                                                                         jint kJ, jobject methodParamsJ)
{
    try {
        return knn_jni::svs_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ,
   jlongArray filteredIdsJ, jint filterIdsTypeJ) {

    try {
        return knn_jni::svs_wrapper::QueryIndex_WithFilter(
            &jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_free(JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        knn_jni::svs_wrapper::Free(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_initLibrary(JNIEnv * env, jclass cls)
{
    try {
        knn_jni::svs_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_sandbox_svs_SvsService_setMergeInterruptCallback(JNIEnv * env, jclass cls)
{
    try {
        faiss::InterruptCallback::instance.reset(
            new knn_jni::svs_wrapper::OpenSearchMergeInterruptCallback(&jniUtil)
        );
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}
