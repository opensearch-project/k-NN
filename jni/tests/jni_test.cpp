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


TEST(GetJavaIntArrayLengthTest, BasicAssertions) {
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();
    JNIEnv_ jniEnv = {jnini};

    auto * vect = new std::vector<int>();
    vect->push_back(1);
    vect->push_back(1);
    vect->push_back(1);
    vect->push_back(1);

    auto * javaIntArray = reinterpret_cast<jintArray> (vect);

    ASSERT_EQ(vect->size(), knn_jni::GetJavaIntArrayLength(&jniEnv, javaIntArray));

    delete vect;
    delete jnini;
}

TEST(ConvertJavaMapToCppMapTest, BasicAssertions) {
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();

    // Override calls that require different return values per call
    static const char * key1 = "key1";
    static const char * key2 = "key2";
    int * value1 = new int;
    *value1 = 10;
    const char * value2 = "value2";
    static const std::vector<mock_jni::MockParameter> mockParams{
            {key1, {mock_jni::TypeMockParameterValue::TYPE_INT, (void*) value1}},
            {key2, {mock_jni::TypeMockParameterValue::TYPE_STRING, (void*) value2}},
    };

    static int callBooleanMethodOrdinal = mockParams.size();
    auto MockCallBooleanMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        // Loop through the vector until there is nothing left and then return false
        if (callBooleanMethodOrdinal-- <= 0) {
            return false;
        }
        return true;
    };

    static long callObjectMethodOrdinal = 0;
    auto MockCallObjectMethod = [](JNIEnv_ jniEnv, jobject obj, jmethodID methodID, va_list args) {
        if ((callObjectMethodOrdinal - 4) % 3 == 0) {
            return (jobject) &(mockParams[(callObjectMethodOrdinal++ - 4) / 3].value);
        }

        if ((callObjectMethodOrdinal - 3) % 3 == 0) {
            return (jobject) mockParams[(callObjectMethodOrdinal++ - 3) / 3].key;
        }

        return (jobject) callObjectMethodOrdinal++; // value does not matter
    };

    jnini->CallObjectMethodV =  reinterpret_cast<jobject (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallObjectMethod);
    jnini->CallBooleanMethodV = reinterpret_cast<jboolean (*)(JNIEnv *, jobject, jmethodID, va_list)>(*MockCallBooleanMethod);

    JNIEnv_ jniEnv = {jnini};

    // We dont actually do anything with parametersJ, the mocking controls the return values
    auto convertedMap = knn_jni::ConvertJavaMapToCppMap(&jniEnv, (jobject) 1);

    ASSERT_EQ(*value1, *(int *)((mock_jni::MockParameterValue *) convertedMap.find(key1)->second)->value);
    ASSERT_EQ(value2, (const char *)((mock_jni::MockParameterValue *) convertedMap.find(key2)->second)->value);

    delete jnini;
    delete value1;
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

TEST(FaissCreateIndex, BasicAssertions) {
    // This will require a lot of mocking - lets go step by step

    // First, create a basic jnienv
    auto * jnini = mock_jni::GenerateMockJNINativeInterface();
    JNIEnv_ jniEnv = {jnini};

    // Next, we need to build parameter map containing all of the necessary values
    //TODO: Is there a way we can just pass the map to a function that will generate the necessary lambda? That would
    // make it much easier to work with parameters. We could even pass the static variable
    static const char * key1 = "key1";
    static const char * key2 = "key2";
    int * value1 = new int;
    *value1 = 10;
    const char * value2 = "value2";
    static const std::vector<mock_jni::MockParameter> mockParams{
            {key1, {mock_jni::TypeMockParameterValue::TYPE_INT, (void*) value1}},
            {key2, {mock_jni::TypeMockParameterValue::TYPE_STRING, (void*) value2}},
    };
}