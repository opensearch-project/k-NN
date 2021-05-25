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

#include "org_opensearch_knn_index_JNIService.h"
#include "jni_util.h"
#include "faiss_wrapper.h"
#include "nmslib_wrapper.h"

#include <jni.h>
#include <string>

JNIEXPORT void JNICALL Java_org_opensearch_knn_index_JNIService_createIndex
        (JNIEnv * env, jclass cls, jintArray idsJ, jobjectArray vectorsJ, jstring indexPathJ, jobject parametersJ,
        jstring engineNameJ) {
    try {
        if (engineNameJ == nullptr) {
            throw std::runtime_error("Engine name cannot be null");
        }

        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);
        if (engineNameCpp == knn_jni::FAISS_NAME) {
            knn_jni::faiss_wrapper::CreateIndex(env, idsJ, vectorsJ, indexPathJ, parametersJ);
            return;
        }

        if (engineNameCpp == knn_jni::NMSLIB_NAME) {
            knn_jni::nmslib_wrapper::CreateIndex(env, idsJ, vectorsJ, indexPathJ, parametersJ);
            return;
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_index_JNIService_createIndexFromTemplate
        (JNIEnv * env, jclass cls, jintArray idsJ, jobjectArray vectorsJ, jstring indexPathJ, jbyteArray templateIndexJ,
         jstring engineNameJ) {
    try {
        if (engineNameJ == nullptr) {
            throw std::runtime_error("Engine name cannot be null");
        }

        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);
        if (engineNameCpp == knn_jni::FAISS_NAME) {
            knn_jni::faiss_wrapper::CreateIndexFromTemplate(env, idsJ, vectorsJ, indexPathJ, templateIndexJ);
            return;
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_index_JNIService_loadIndex
        (JNIEnv * env, jclass cls, jstring indexPathJ, jobject parametersJ, jstring engineNameJ) {
    try {
        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);

        if (engineNameCpp == knn_jni::FAISS_NAME) {
            return knn_jni::faiss_wrapper::LoadIndex(env, indexPathJ);
        }

        if (engineNameCpp == knn_jni::NMSLIB_NAME) {
            return knn_jni::nmslib_wrapper::LoadIndex(env, indexPathJ, parametersJ);
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_index_JNIService_queryIndex
        (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject parametersJ,
         jstring engineNameJ) {
    try {
        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);

        if (engineNameCpp == knn_jni::FAISS_NAME) {
            return knn_jni::faiss_wrapper::QueryIndex(env, indexPointerJ, queryVectorJ, kJ);
        }

        if (engineNameCpp == knn_jni::NMSLIB_NAME) {
            return knn_jni::nmslib_wrapper::QueryIndex(env, indexPointerJ, queryVectorJ, kJ);
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }

    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_index_JNIService_free
(JNIEnv * env, jclass cls, jlong indexPointerJ, jstring engineNameJ) {
    try {
        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);

        if (engineNameCpp == knn_jni::FAISS_NAME) {
            return knn_jni::faiss_wrapper::Free(indexPointerJ);
        }

        if (engineNameCpp == knn_jni::NMSLIB_NAME) {
            return knn_jni::nmslib_wrapper::Free(indexPointerJ);
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_index_JNIService_initLibrary
(JNIEnv * env, jclass cls, jstring engineNameJ) {
    try {
        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);

        if (engineNameCpp == knn_jni::FAISS_NAME) {
            knn_jni::faiss_wrapper::InitLibrary();
            return;
        }

        if (engineNameCpp == knn_jni::NMSLIB_NAME) {
            knn_jni::nmslib_wrapper::InitLibrary();
            return;
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_index_JNIService_trainIndex
        (JNIEnv * env, jclass cls, jobject parametersJ, jint dimension, jlong trainVectorsPointerJ,
         jstring engineNameJ) {
    try {
        std::string engineNameCpp = knn_jni::ConvertJavaStringToCppString(env, engineNameJ);

        if (engineNameCpp == knn_jni::FAISS_NAME) {
            return knn_jni::faiss_wrapper::TrainIndex(env, parametersJ, dimension, trainVectorsPointerJ);
        }

        knn_jni::ThrowJavaException(env, knn_jni::ILLEGAL_ARGUMENT_PATH.c_str(), "Invalid engine");
    } catch (...) {
        knn_jni::CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_index_JNIService_transferVectors
        (JNIEnv * env, jclass cls, jlong vectorsPointerJ, jobjectArray vectorsJ) {
    std::vector<float> *vect;
    if ((long) vectorsPointerJ == 0) {
        vect = new std::vector<float>;
    } else {
        vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
    }

    int dim = knn_jni::GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = knn_jni::Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);
    vect->insert(vect->begin(), dataset.begin(), dataset.end());

    return (jlong) vect;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_index_JNIService_freeVectors
        (JNIEnv * env, jclass cls, jlong vectorsPointerJ) {
    if (vectorsPointerJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
        delete vect;
    }
}
