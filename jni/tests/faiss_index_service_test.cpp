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


#include "faiss_index_service.h"
#include "mocks/faiss_methods_mock.h"
#include "mocks/faiss_index_mock.h"
#include "test_util.h"
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "commons.h"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

TEST(CreateIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<int64_t> ids;
    std::vector<float> vectors;
    int dim = 2;
    vectors.reserve(dim * numIds);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "HNSW32,Flat";
    int threadCount = 1;
    std::unordered_map<std::string, jobject> parametersMap;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Setup faiss method mock
    // This object is handled by unique_ptr inside indexService.createIndex()
    MockIndex* index = new MockIndex();
    EXPECT_CALL(*index, add(numIds, vectors.data()))
        .Times(1);
    // This object is handled by unique_ptr inside indexService.createIndex()
    faiss::IndexIDMap* indexIdMap = new faiss::IndexIDMap(index);
    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    EXPECT_CALL(*mockFaissMethods, indexFactory(dim, ::testing::StrEq(indexDescription.c_str()), metricType))
        .WillOnce(Return(index));
    EXPECT_CALL(*mockFaissMethods, indexIdMap(index))
        .WillOnce(Return(indexIdMap));
    EXPECT_CALL(*mockFaissMethods, writeIndex(indexIdMap, ::testing::StrEq(indexPath.c_str())))
        .Times(1);

    // Create the index
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));
    indexService.createIndex(
        &mockJNIUtil,
        jniEnv,
        metricType,
        indexDescription,
        dim,
        numIds,
        threadCount,
        (int64_t) &vectors,
        ids,
        indexPath,
        parametersMap);
}

TEST(CreateBinaryIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<uint8_t> vectors;
    int dim = 128;
    vectors.reserve(numIds);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "BHNSW32";
    int threadCount = 1;
    std::unordered_map<std::string, jobject> parametersMap;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Setup faiss method mock
    // This object is handled by unique_ptr inside indexService.createIndex()
    MockIndexBinary* index = new MockIndexBinary();
    EXPECT_CALL(*index, add(numIds, vectors.data()))
        .Times(1);
    // This object is handled by unique_ptr inside indexService.createIndex()
    faiss::IndexBinaryIDMap* indexIdMap = new faiss::IndexBinaryIDMap(index);
    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    EXPECT_CALL(*mockFaissMethods, indexBinaryFactory(dim, ::testing::StrEq(indexDescription.c_str())))
        .WillOnce(Return(index));
    EXPECT_CALL(*mockFaissMethods, indexBinaryIdMap(index))
        .WillOnce(Return(indexIdMap));
    EXPECT_CALL(*mockFaissMethods, writeIndexBinary(indexIdMap, ::testing::StrEq(indexPath.c_str())))
        .Times(1);

    // Create the index
    knn_jni::faiss_wrapper::BinaryIndexService indexService(std::move(mockFaissMethods));
    indexService.createIndex(
        &mockJNIUtil,
        jniEnv,
        metricType,
        indexDescription,
        dim,
        numIds,
        threadCount,
        (int64_t) &vectors,
        ids,
        indexPath,
        parametersMap);
}