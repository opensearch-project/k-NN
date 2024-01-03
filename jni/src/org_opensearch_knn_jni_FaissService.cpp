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

#include "org_opensearch_knn_jni_FaissService.h"

#include <jni.h>

#include <algorithm>
#include <vector>

#include "faiss_wrapper.h"
#include "jni_util.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_FAISS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env);

    return KNN_FAISS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION);
    jniUtil.Uninitialize(env);
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                            jobjectArray vectorsJ, jstring indexPathJ,
                                                                            jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndex(&jniUtil, env, idsJ, vectorsJ, indexPathJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(JNIEnv * env, jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jobjectArray vectorsJ,
                                                                                        jstring indexPathJ,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndexFromTemplate(&jniUtil, env, idsJ, vectorsJ, indexPathJ, templateIndexJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
    try {
        return knn_jni::faiss_wrapper::LoadIndex(&jniUtil, env, indexPathJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ, jint kJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, parentIdsJ);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jintArray filteredIdsJ, jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, filteredIdsJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free(JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::Free(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary(JNIEnv * env, jclass cls)
{
    try {
        knn_jni::faiss_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors(JNIEnv * env, jclass cls,
                                                                                 jlong vectorsPointerJ,
                                                                                 jobjectArray vectorsJ)
{
    std::vector<float> *vect;
    if ((long) vectorsPointerJ == 0) {
        vect = new std::vector<float>;
    } else {
        vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
    }

    int dim = jniUtil.GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil.Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);
    vect->insert(vect->begin(), dataset.begin(), dataset.end());

    return (jlong) vect;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeVectors(JNIEnv * env, jclass cls,
                                                                            jlong vectorsPointerJ)
{
    if (vectorsPointerJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
        delete vect;
    }
}
