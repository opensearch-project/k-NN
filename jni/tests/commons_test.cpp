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


#include "test_util.h"
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "commons.h"

TEST(CommonsTests, BasicAssertions) {
    long dim = 3;
    long totalNumberOfVector = 5;
    std::vector<std::vector<float>> data;
    for(int i = 0 ; i < totalNumberOfVector - 1 ; i++) {
        std::vector<float> vector;
        for(int j = 0 ; j < dim ; j ++) {
            vector.push_back((float)j);
        }
        data.push_back(vector);
    }
    JNIEnv *jniEnv = nullptr;

    testing::NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    jlong memoryAddress = knn_jni::commons::storeVectorData(&mockJNIUtil, jniEnv, (jlong)0,
                      reinterpret_cast<jobjectArray>(&data), (jlong)(totalNumberOfVector * dim));
    ASSERT_NE(memoryAddress, 0);
    auto *vect = reinterpret_cast<std::vector<float>*>(memoryAddress);
    ASSERT_EQ(vect->size(), data.size() * dim);
    ASSERT_EQ(vect->capacity(), totalNumberOfVector * dim);

    // Check by inserting more vectors at same memory location
    jlong oldMemoryAddress = memoryAddress;
    std::vector<std::vector<float>> data2;
    std::vector<float> vector;
    for(int j = 0 ; j < dim ; j ++) {
        vector.push_back((float)j);
    }
    data2.push_back(vector);
    memoryAddress = knn_jni::commons::storeVectorData(&mockJNIUtil, jniEnv, memoryAddress,
        reinterpret_cast<jobjectArray>(&data2), (jlong)(totalNumberOfVector * dim));
    ASSERT_NE(memoryAddress, 0);
    ASSERT_EQ(memoryAddress, oldMemoryAddress);
    vect = reinterpret_cast<std::vector<float>*>(memoryAddress);
    int currentIndex = 0;
    ASSERT_EQ(vect->size(), totalNumberOfVector*dim);
    ASSERT_EQ(vect->capacity(), totalNumberOfVector * dim);

    // Validate if all vectors data are at correct location
    for(auto & i : data) {
        for(float j : i) {
            ASSERT_FLOAT_EQ(vect->at(currentIndex), j);
            currentIndex++;
        }
    }

    for(auto & i : data2) {
        for(float j : i) {
            ASSERT_FLOAT_EQ(vect->at(currentIndex), j);
            currentIndex++;
        }
    }
}

TEST(CommonTests, GetIntegerMethodParam) {
    JNIEnv *jniEnv = nullptr;
    testing::NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unordered_map<std::string, jobject> methodParams1;
    int efSearch = 10;
    methodParams1[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    int actualValue1 = knn_jni::commons::getIntegerMethodParameter(jniEnv, &mockJNIUtil, methodParams1, knn_jni::EF_SEARCH, 1);
    EXPECT_EQ(efSearch, actualValue1);

    int actualValue2 = knn_jni::commons::getIntegerMethodParameter(jniEnv, &mockJNIUtil, methodParams1, "param", 1);
    EXPECT_EQ(1, actualValue2);

    std::unordered_map<std::string, jobject> methodParams2;
    int actualValue3 = knn_jni::commons::getIntegerMethodParameter(jniEnv, &mockJNIUtil, methodParams2, knn_jni::EF_SEARCH, 1);
    EXPECT_EQ(1, actualValue3);
}
