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
                                                                            jlong vectorsAddressJ, jint dimJ,
                                                                            jstring indexPathJ, jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::CreateIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexPathJ, parametersJ, &indexService);

        // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
        // This is not the ideal approach, please refer this gh issue for long term solution:
        // https://github.com/opensearch-project/k-NN/issues/1600
        delete reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                            jlong vectorsAddressJ, jint dimJ,
                                                                            jstring indexPathJ, jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::CreateIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexPathJ, parametersJ, &binaryIndexService);

        // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
        // This is not the ideal approach, please refer this gh issue for long term solution:
        // https://github.com/opensearch-project/k-NN/issues/1600
        delete reinterpret_cast<std::vector<uint8_t>*>(vectorsAddressJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(JNIEnv * env, jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jlong vectorsAddressJ,
                                                                                        jint dimJ,
                                                                                        jstring indexPathJ,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndexFromTemplate(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexPathJ, templateIndexJ, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate(JNIEnv * env, jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jlong vectorsAddressJ,
                                                                                        jint dimJ,
                                                                                        jstring indexPathJ,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateBinaryIndexFromTemplate(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexPathJ, templateIndexJ, parametersJ);
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

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
    try {
        return knn_jni::faiss_wrapper::LoadBinaryIndex(&jniUtil, env, indexPathJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired
        (JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::IsSharedIndexStateRequired(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::InitSharedIndexState(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::SetSharedIndexState(indexPointerJ, shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, parentIdsJ);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jbyteArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free(JNIEnv * env, jclass cls, jlong indexPointerJ, jboolean isBinaryIndexJ)
{
    try {
        return knn_jni::faiss_wrapper::Free(indexPointerJ, isBinaryIndexJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
        (JNIEnv * env, jclass cls, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::FreeSharedIndexState(shareIndexStatePointerJ);
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

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainBinaryIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
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

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ,
                                                                                   jfloat radiusJ, jobject methodParamsJ,
                                                                                   jint maxResultWindowJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearch(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ,
                                                                                   jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ,
                                                                                   jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearchWithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, filterIdsJ, filterIdsTypeJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}
