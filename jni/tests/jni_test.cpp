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

#include "jni_util.h"
#include "faiss_wrapper.h"
#include "nmslib_wrapper.h"
#include "mock_jni.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <org_opensearch_knn_index_JNIService.h>

using ::testing::Return;

// Mock test
// For simple cases, we can implement the behavior automatically. For other cases, we will need to mock
class MockJNIUtil : public knn_jni::JNIUtilInterface {
public:

    MockJNIUtil() {
        // At the moment, these are the defaults. Either make them oncall, or static
        EXPECT_CALL(*this, ConvertJavaObjectToCppString).WillRepeatedly([this](JNIEnv * env, jobject objectJ) {
            return *((std::string *) objectJ);
        });

        EXPECT_CALL(*this, ConvertJavaStringToCppString).WillRepeatedly([this](JNIEnv * env, jstring stringJ) {
            return *((std::string *) stringJ);
        });

        EXPECT_CALL(*this, GetJavaIntArrayLength).WillRepeatedly([this](JNIEnv *env, jintArray arrayJ) {
            return reinterpret_cast<std::vector<int64_t> *>(arrayJ)->size();
        });

        EXPECT_CALL(*this, GetInnerDimensionOf2dJavaFloatArray).WillRepeatedly([this](JNIEnv *env, jobjectArray array2dJ) {
            return (*reinterpret_cast<std::vector<std::vector<float>> *>(array2dJ))[0].size();
        });

        EXPECT_CALL(*this, Convert2dJavaObjectArrayToCppFloatVector).WillRepeatedly([this](JNIEnv *env, jobjectArray array2dJ, int dim) {
            std::vector<float> data;
            for (auto v : (*reinterpret_cast<std::vector<std::vector<float>> *>(array2dJ)))
                for (auto item : v)
                    data.push_back(item);
            return data;
        });

        EXPECT_CALL(*this, ConvertJavaIntArrayToCppIntVector).WillRepeatedly([this](JNIEnv *env, jintArray arrayJ) {
            return *reinterpret_cast<std::vector<int64_t> *>(arrayJ);
        });

        EXPECT_CALL(*this, ConvertJavaMapToCppMap).WillRepeatedly([this](JNIEnv * env, jobject parametersJ) {
            return *reinterpret_cast<std::unordered_map<std::string, jobject> *>(parametersJ);
        });

        EXPECT_CALL(*this, DeleteLocalRef).WillRepeatedly([this](JNIEnv *env, jobject obj) {});
    }

    MOCK_METHOD(void, ThrowJavaException, (JNIEnv* env, const char* type, const char* message));
    MOCK_METHOD(void, HasExceptionInStack, (JNIEnv* env));
    MOCK_METHOD(void, HasExceptionInStack, (JNIEnv* env, const std::string& message));
    MOCK_METHOD(void, CatchCppExceptionAndThrowJava, (JNIEnv* env));
    MOCK_METHOD(jclass, FindClass, (JNIEnv * env, const std::string& className));
    MOCK_METHOD(jmethodID, FindMethod, (JNIEnv * env, jclass jClass, const std::string& methodName, const std::string& methodSignature));
    MOCK_METHOD(std::string, ConvertJavaStringToCppString, (JNIEnv * env, jstring javaString));
    //TODO: Figure out why this cant use "new" MOCK_METHOD
    MOCK_METHOD2(ConvertJavaMapToCppMap, std::unordered_map<std::string, jobject>(JNIEnv * env, jobject parametersJ));
    MOCK_METHOD(std::string, ConvertJavaObjectToCppString, (JNIEnv * env, jobject objectJ));
    MOCK_METHOD(int, ConvertJavaObjectToCppInteger, (JNIEnv *env, jobject objectJ));
    MOCK_METHOD(std::vector<float>, Convert2dJavaObjectArrayToCppFloatVector, (JNIEnv *env, jobjectArray array2dJ, int dim));
    MOCK_METHOD(std::vector<int64_t>, ConvertJavaIntArrayToCppIntVector, (JNIEnv *env, jintArray arrayJ));
    MOCK_METHOD(int, GetInnerDimensionOf2dJavaFloatArray, (JNIEnv *env, jobjectArray array2dJ));
    MOCK_METHOD(int, GetJavaObjectArrayLength, (JNIEnv *env, jobjectArray arrayJ));
    MOCK_METHOD(int, GetJavaIntArrayLength, (JNIEnv *env, jintArray arrayJ));
    MOCK_METHOD(int, GetJavaBytesArrayLength, (JNIEnv *env, jbyteArray arrayJ));
    MOCK_METHOD(int, GetJavaFloatArrayLength, (JNIEnv *env, jfloatArray arrayJ));
    MOCK_METHOD(void, DeleteLocalRef, (JNIEnv *env, jobject obj));
    MOCK_METHOD(jbyte *, GetByteArrayElements, (JNIEnv *env, jbyteArray array, jboolean * isCopy));
    MOCK_METHOD(jfloat *, GetFloatArrayElements, (JNIEnv *env, jfloatArray array, jboolean * isCopy));
    MOCK_METHOD(jint *, GetIntArrayElements, (JNIEnv *env, jintArray array, jboolean * isCopy));
    MOCK_METHOD(jobject, GetObjectArrayElement, (JNIEnv *env, jobjectArray array, jsize index));
    MOCK_METHOD(jobject, NewObject, (JNIEnv *env, jclass clazz, jmethodID methodId, int id, float distance));
    MOCK_METHOD(jobjectArray, NewObjectArray, (JNIEnv *env, jsize len, jclass clazz, jobject init));
    MOCK_METHOD(jbyteArray, NewByteArray, (JNIEnv *env, jsize len));
    MOCK_METHOD(void, ReleaseByteArrayElements, (JNIEnv *env, jbyteArray array, jbyte *elems, int mode));
    MOCK_METHOD(void, ReleaseFloatArrayElements, (JNIEnv *env, jfloatArray array, jfloat *elems, int mode));
    MOCK_METHOD(void, ReleaseIntArrayElements, (JNIEnv *env, jintArray array, jint *elems, jint mode));
    MOCK_METHOD(void, SetObjectArrayElement, (JNIEnv *env, jobjectArray array, jsize index, jobject val));
    MOCK_METHOD(void, SetByteArrayRegion, (JNIEnv *env, jbyteArray array, jsize start, jsize len, const jbyte * buf));
};

TEST(FaissCreateIndexTest, BasicAssertions) {
    // Define the data
    JNIEnv * jniEnv = nullptr;

    int64_t idsnum = 100;
    std::vector<int64_t> ids;
    std::vector<std::vector<float>> vectors;
    for (int64_t i = 0; i < idsnum; ++i) {
        ids.push_back(i);
        vectors.push_back({1.0, 2.0});
    }

    std::string indexPath = "test-create.faiss";
    std::string spaceType = knn_jni::L2;
    std::string method = "HNSW32,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
    parametersMap[knn_jni::METHOD] = (jobject) &method;

    // Set up the jni mocking
    MockJNIUtil mockJNIUtil;

    EXPECT_CALL(mockJNIUtil, GetJavaObjectArrayLength(jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

    // Create the index
    knn_jni::faiss_wrapper::CreateIndex(&mockJNIUtil,
                                        jniEnv,
                                        reinterpret_cast<jintArray>(&ids),
                                        reinterpret_cast<jobjectArray>(&vectors),
                                        (jstring) &indexPath,
                                        (jobject) &parametersMap);
}

TEST(InitLibraryTest, BasicAssertions) {
    knn_jni::faiss_wrapper::InitLibrary();
    knn_jni::nmslib_wrapper::InitLibrary();
}

TEST(FreeVectorsTest, BasicAssertions) {
    auto *vect = new std::vector<float>;
    vect->push_back(1.0);
    vect->push_back(2.0);
    vect->push_back(3.0);

    Java_org_opensearch_knn_index_JNIService_freeVectors(nullptr, nullptr, reinterpret_cast<jlong>(vect));
}