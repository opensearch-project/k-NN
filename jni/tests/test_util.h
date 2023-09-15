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

#ifndef OPENSEARCH_KNN_TEST_UTIL_H
#define OPENSEARCH_KNN_TEST_UTIL_H

#include <gmock/gmock.h>
#include <jni.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "faiss/Index.h"
#include "faiss/MetaIndexes.h"
#include "faiss/MetricType.h"
#include "faiss/impl/io.h"
#include "index.h"
#include "init.h"
#include "jni_util.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "params.h"
#include "space.h"
#include "spacefactory.h"

namespace test_util {
    class MockJNIUtil : public knn_jni::JNIUtilInterface {
    public:
        MockJNIUtil();
        MOCK_METHOD(void, CatchCppExceptionAndThrowJava, (JNIEnv * env));
        MOCK_METHOD(std::string, ConvertJavaStringToCppString,
                    (JNIEnv * env, jstring javaString));
        // TODO: Figure out why this cant use "new" MOCK_METHOD
        MOCK_METHOD(std::vector<float>, Convert2dJavaObjectArrayToCppFloatVector,
                    (JNIEnv * env, jobjectArray array2dJ, int dim));
        MOCK_METHOD(std::vector<int64_t>, ConvertJavaIntArrayToCppIntVector,
                    (JNIEnv * env, jintArray arrayJ));
        MOCK_METHOD2(ConvertJavaMapToCppMap,
                     std::unordered_map<std::string, jobject>(JNIEnv* env,
                                                              jobject parametersJ));
        MOCK_METHOD(int, ConvertJavaObjectToCppInteger,
                    (JNIEnv * env, jobject objectJ));
        MOCK_METHOD(std::string, ConvertJavaObjectToCppString,
                    (JNIEnv * env, jobject objectJ));
        MOCK_METHOD(void, DeleteLocalRef, (JNIEnv * env, jobject obj));
        MOCK_METHOD(jclass, FindClass, (JNIEnv * env, const std::string& className));
        MOCK_METHOD(jmethodID, FindMethod, (JNIEnv * env, const std::string& className, const std::string& methodName));
        MOCK_METHOD(jbyte*, GetByteArrayElements,
                    (JNIEnv * env, jbyteArray array, jboolean* isCopy));
        MOCK_METHOD(jfloat*, GetFloatArrayElements,
                    (JNIEnv * env, jfloatArray array, jboolean* isCopy));
        MOCK_METHOD(int, GetInnerDimensionOf2dJavaFloatArray,
                    (JNIEnv * env, jobjectArray array2dJ));
        MOCK_METHOD(jint*, GetIntArrayElements,
                    (JNIEnv * env, jintArray array, jboolean* isCopy));
        MOCK_METHOD(int, GetJavaBytesArrayLength, (JNIEnv * env, jbyteArray arrayJ));
        MOCK_METHOD(int, GetJavaFloatArrayLength, (JNIEnv * env, jfloatArray arrayJ));
        MOCK_METHOD(int, GetJavaIntArrayLength, (JNIEnv * env, jintArray arrayJ));
        MOCK_METHOD(int, GetJavaObjectArrayLength,
                    (JNIEnv * env, jobjectArray arrayJ));
        MOCK_METHOD(jobject, GetObjectArrayElement,
                    (JNIEnv * env, jobjectArray arrayJ, jsize index));
        MOCK_METHOD(void, HasExceptionInStack, (JNIEnv * env));
        MOCK_METHOD(void, HasExceptionInStack,
                    (JNIEnv * env, const std::string& message));
        MOCK_METHOD(jbyteArray, NewByteArray, (JNIEnv * env, jsize len));
        MOCK_METHOD(jobject, NewObject,
                    (JNIEnv * env, jclass clazz, jmethodID methodId, int id,
                            float distance));
        MOCK_METHOD(jobjectArray, NewObjectArray,
                    (JNIEnv * env, jsize len, jclass clazz, jobject init));
        MOCK_METHOD(void, ReleaseByteArrayElements,
                    (JNIEnv * env, jbyteArray array, jbyte* elems, int mode));
        MOCK_METHOD(void, ReleaseFloatArrayElements,
                    (JNIEnv * env, jfloatArray array, jfloat* elems, int mode));
        MOCK_METHOD(void, ReleaseIntArrayElements,
                    (JNIEnv * env, jintArray array, jint* elems, jint mode));
        MOCK_METHOD(void, SetByteArrayRegion,
                    (JNIEnv * env, jbyteArray array, jsize start, jsize len,
                            const jbyte* buf));
        MOCK_METHOD(void, SetObjectArrayElement,
                    (JNIEnv * env, jobjectArray array, jsize index, jobject val));
        MOCK_METHOD(void, ThrowJavaException,
                    (JNIEnv * env, const char* type, const char* message));
    };

// For our unit tests, we want to ensure that each test tests one function in
// isolation. So, we add a few utils to perform common library operations

// -------------------------------- FAISS UTILS ----------------------------------

    faiss::Index* FaissCreateIndex(int dim, const std::string& method,
                                   faiss::MetricType metric);

    faiss::VectorIOWriter FaissGetSerializedIndex(faiss::Index* index);

    faiss::Index* FaissLoadFromSerializedIndex(std::vector<uint8_t>* indexSerial);

    faiss::IndexIDMap FaissAddData(faiss::Index* index,
                                   std::vector<faiss::idx_t> ids,
                                   std::vector<float> dataset);

    void FaissWriteIndex(faiss::Index* index, const std::string& indexPath);

    faiss::Index* FaissLoadIndex(const std::string& indexPath);

    void FaissQueryIndex(faiss::Index* index, float* query, int k, float* distances,
                         faiss::idx_t* ids);

    void FaissTrainIndex(faiss::Index* index, faiss::idx_t n,
                         const float* x);

// -------------------------------------------------------------------------------

// ------------------------------- NMSLIB UTILS ----------------------------------

    similarity::Index<float>* NmslibCreateIndex(
            int* ids, std::vector<std::vector<float>> dataset,
            similarity::Space<float>* space, const std::string& spaceName,
            const std::vector<std::string>& indexParameters);

    void NmslibWriteIndex(similarity::Index<float>* index,
                          const std::string& indexPath);

    similarity::Index<float>* NmslibLoadIndex(
            const std::string& indexPath, similarity::Space<float>* space,
            const std::string& spaceName,
            const std::vector<std::string>& queryParameters);

    similarity::KNNQuery<float>* NmslibQueryIndex(similarity::Index<float>* index,
                                                  float* query, int k, int dim,
                                                  similarity::Space<float>* space);

// -------------------------------------------------------------------------------

// ------------------------------- OTHER UTILS ----------------------------------

    std::string RandomString(size_t length, const std::string& prefix, const std::string& suffix);

    float RandomFloat(float min, float max);

    // Read vector file formats
    void load_data(char* filename, float*& data, unsigned& num, unsigned& dim);

// -------------------------------------------------------------------------------
}  // namespace test_util

#endif  // OPENSEARCH_KNN_TEST_UTIL_H
