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
#include <vector>
#include <org_opensearch_knn_index_JNIService.h>

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

TEST(GetJavaIntArrayLengthTest, BasicAssertions) {
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();
    JNIEnv_ jniEnv = {jnini};

    std::vector<int> vect = {
            1, 2, 3, 4
    };

    auto * javaIntArray = reinterpret_cast<jintArray> (&vect);

    ASSERT_EQ(vect.size(), knn_jni::GetJavaIntArrayLength(&jniEnv, javaIntArray));

    delete jnini;
}

TEST(ConvertJavaMapToCppMapTest, BasicAssertions) {
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();

    // Define values of interest for checks
    const char * key1 = "key1";
    const char * key2 = "key2";
    int value1 = 10;
    const char * value2 = "value2";
    const char * dummyValue = "dummy";

    // Vector of returns from MockCallObjectMethod
    static const std::vector<void *> mockObjectCallValues{
            (void*) dummyValue,
            (void*) dummyValue,
            (void*) dummyValue,
            (void*) key1,
            (void*) &value1,
            (void*) dummyValue,
            (void*) key2,
            (void*) value2
    };

    // Initialize static variables that can be used in lambdas
    static int callObjectMethodOrdinal = 0;
    auto MockCallObjectMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        return (jobject) mockObjectCallValues[callObjectMethodOrdinal++];
    };

    static int callBooleanMethodOrdinal = 2;
    auto MockCallBooleanMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        // Loop through the vector until there is nothing left and then return false
        if (callBooleanMethodOrdinal-- <= 0) {
            return false;
        }
        return true;
    };

    jnini->CallObjectMethodV =  reinterpret_cast<jobject (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallObjectMethod);
    jnini->CallBooleanMethodV = reinterpret_cast<jboolean (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallBooleanMethod);

    JNIEnv_ jniEnv = {jnini};

    // We dont actually do anything with parametersJ, the mocking controls the return values
    auto convertedMap = knn_jni::ConvertJavaMapToCppMap(&jniEnv, (jobject) 1);

    ASSERT_EQ(value1, *(int *) convertedMap.find(key1)->second);
    ASSERT_EQ(value2, (const char *) convertedMap.find(key2)->second);

    delete jnini;
}

TEST(FaissCreateIndexTest, BasicAssertions) {
    // Define data
    const int dim = 2;
    const std::vector<int> ids = {
            1, 2, 3, 4
    };

    static const std::vector<std::vector<float>> vectors = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0},
            {7.0, 8.0},
    };

    // Create a basic jnienv
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();

    // Mock GetArrayLength
    static std::vector<int> mockArrayLengthCalls{
        static_cast<int>(vectors.size()),
        static_cast<int>(ids.size()),
        static_cast<int>(vectors.size()),
        dim,
        static_cast<int>(vectors.size()),
    };

    for (auto i : vectors)
        mockArrayLengthCalls.push_back(i.size());

    mockArrayLengthCalls.push_back(static_cast<int>(ids.size()));

    static int getArrayLengthOrdinal = 0;
    auto MockGetArrayLength = [](JNIEnv_ jniEnv, jintArray array) {
        return mockArrayLengthCalls[getArrayLengthOrdinal++];
    };

    // Mock GetObjectArrayElement
    static long getObjectArrayElementOrdinal = 0;
    // can this be simplified to use array instead of static vector?
    auto MockGetObjectArrayElement = [](JNIEnv_ jniEnv, jobjectArray array, int index) {
        if (getObjectArrayElementOrdinal == 0) {
            return (jobject) ++getObjectArrayElementOrdinal;
        }
        return (jobject) &vectors[getObjectArrayElementOrdinal++ - 1];
    };

    // Define values returned by CallObject
    const char * dummyValue = "dummy";
    static const std::vector<void *> mockObjectCallValues{
            (void*) dummyValue,
            (void*) dummyValue,
            (void*) dummyValue,
            (void*) knn_jni::SPACE_TYPE.c_str(),
            (void*) knn_jni::L2.c_str(),
            (void*) dummyValue,
            (void*) knn_jni::METHOD.c_str(),
            (void*) "HNSW32,Flat"
    };

    // Mock CallObject
    static int callObjectMethodOrdinal = 0;
    auto MockCallObjectMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        return (jobject) mockObjectCallValues[callObjectMethodOrdinal++];
    };

    // Mock CallBooleanMethod
    static int callBooleanMethodOrdinal = 2;
    auto MockCallBooleanMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        // Loop through the vector until there is nothing left and then return false
        if (callBooleanMethodOrdinal-- <= 0) {
            return false;
        }
        return true;
    };

    jnini->GetArrayLength = reinterpret_cast<jsize (*)(JNIEnv *, jarray)>(*MockGetArrayLength);
    jnini->CallObjectMethodV =  reinterpret_cast<jobject (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallObjectMethod);
    jnini->CallBooleanMethodV = reinterpret_cast<jboolean (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallBooleanMethod);
    jnini->GetObjectArrayElement = reinterpret_cast<jobject (*)(JNIEnv *, jobjectArray, int)>(*MockGetObjectArrayElement);

    JNIEnv_ jniEnv = {jnini};

    const char * tempPath = "test22.faiss";

    knn_jni::faiss_wrapper::CreateIndex(&jniEnv, (jintArray) &ids, (jobjectArray) &vectors, (jstring) tempPath, (jobject) 1);
}