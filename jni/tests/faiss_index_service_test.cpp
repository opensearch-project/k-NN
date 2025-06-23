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
#include <faiss/IndexFlat.h>

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::_;

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
    faiss::FileIOWriter fileIOWriter {indexPath.c_str()};
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
    EXPECT_CALL(*mockFaissMethods, writeIndex(indexIdMap, ::testing::Eq(&fileIOWriter)))
        .Times(1);

    // Create the index
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));
    long indexAddress = indexService.initIndex(&mockJNIUtil, jniEnv, metricType, indexDescription, dim, numIds, threadCount, parametersMap);
    indexService.insertToIndex(dim, numIds, threadCount, (int64_t) &vectors, ids, indexAddress);
    indexService.writeIndex(&fileIOWriter, indexAddress);
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
    faiss::FileIOWriter fileIOWriter {indexPath.c_str()};
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
    EXPECT_CALL(*mockFaissMethods, writeIndexBinary(indexIdMap, ::testing::Eq(&fileIOWriter)))
        .Times(1);

    // Create the index
    knn_jni::faiss_wrapper::BinaryIndexService indexService(std::move(mockFaissMethods));
    long indexAddress = indexService.initIndex(&mockJNIUtil, jniEnv, metricType, indexDescription, dim, numIds, threadCount, parametersMap);
    indexService.insertToIndex(dim, numIds, threadCount, (int64_t) &vectors, ids, indexAddress);
    indexService.writeIndex(&fileIOWriter, indexAddress);
}

TEST(CreateByteIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<int8_t> vectors;
    int dim = 8;
    vectors.reserve(numIds * dim);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
            vectors.push_back(test_util::RandomInt(-128, 127));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::FileIOWriter fileIOWriter {indexPath.c_str()};
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "HNSW16,SQ8_direct_signed";
    int threadCount = 1;
    std::unordered_map<std::string, jobject> parametersMap;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Setup faiss method mock
    // This object is handled by unique_ptr inside indexService.createIndex()
    MockIndex* index = new MockIndex();
    // This object is handled by unique_ptr inside indexService.createIndex()
    faiss::IndexIDMap* indexIdMap = new faiss::IndexIDMap(index);
    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    EXPECT_CALL(*mockFaissMethods, indexFactory(dim, ::testing::StrEq(indexDescription.c_str()), metricType))
        .WillOnce(Return(index));
    EXPECT_CALL(*mockFaissMethods, indexIdMap(index))
        .WillOnce(Return(indexIdMap));
    EXPECT_CALL(*mockFaissMethods, writeIndex(indexIdMap, ::testing::Eq(&fileIOWriter)))
        .Times(1);

    // Create the index
    knn_jni::faiss_wrapper::ByteIndexService indexService(std::move(mockFaissMethods));
    long indexAddress = indexService.initIndex(&mockJNIUtil, jniEnv, metricType, indexDescription, dim, numIds, threadCount, parametersMap);
    indexService.insertToIndex(dim, numIds, threadCount, (int64_t) &vectors, ids, indexAddress);
    indexService.writeIndex(&fileIOWriter, indexAddress);
}

//buildFlatIndexFromVectors tests

// Helper: Create dummy data for float vectors
std::vector<float> makeVectors(int num, int dim, float val = 1.0f) {
    std::vector<float> v(num * dim, val);
    return v;
}

/**
 * Test that a Flat L2 index is successfully created from vectors.
 * Checks that the returned pointer is not null and the number of vectors is correct.
 */
TEST(BuildFlatIndexFromVectorsTest, BuildsL2Index) {
    int numVectors = 5, dim = 3;
    std::vector<float> data = makeVectors(numVectors, dim, 2.0f);

    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));

    jlong indexPtr = indexService.buildFlatIndexFromVectors(numVectors, dim, data, faiss::METRIC_L2);

    ASSERT_NE(indexPtr, 0);
    auto* index = reinterpret_cast<faiss::IndexFlatL2*>(indexPtr);
    ASSERT_EQ(index->ntotal, numVectors);
    delete index;
}

/**
 * Test that a Flat Inner Product index is successfully created from vectors.
 * Checks that the returned pointer is not null and the number of vectors is correct.
 */
TEST(BuildFlatIndexFromVectorsTest, BuildsIPIndex) {
    int numVectors = 4, dim = 2;
    std::vector<float> data = makeVectors(numVectors, dim, 3.0f);

    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));

    jlong indexPtr = indexService.buildFlatIndexFromVectors(numVectors, dim, data, faiss::METRIC_INNER_PRODUCT);

    ASSERT_NE(indexPtr, 0);
    auto* index = reinterpret_cast<faiss::IndexFlatIP*>(indexPtr);
    ASSERT_EQ(index->ntotal, numVectors);
    delete index;
}

/**
 * Test that providing empty vectors throws a runtime_error.
 */
TEST(BuildFlatIndexFromVectorsTest, ThrowsOnEmptyVectors) {
    int numVectors = 10, dim = 4;
    std::vector<float> empty;

    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));

    EXPECT_THROW(
        indexService.buildFlatIndexFromVectors(numVectors, dim, empty, faiss::METRIC_L2),
        std::runtime_error
    );
}

/**
 * Test that providing a vector whose size does not match numVectors * dim throws a runtime_error.
 */
TEST(BuildFlatIndexFromVectorsTest, ThrowsOnMismatchedSize) {
    int numVectors = 3, dim = 5;
    std::vector<float> badData(7, 1.0f);

    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));

    EXPECT_THROW(
        indexService.buildFlatIndexFromVectors(numVectors, dim, badData, faiss::METRIC_L2),
        std::runtime_error
    );
}

/**
 * Test that the vectors inserted into the flat index are preserved in order and value.
 * This reconstructs each vector from the index and compares to the original input.
 */
TEST(BuildFlatIndexFromVectorsTest, IndexContainsInsertedVectorsInOrder) {
    int numVectors = 5, dim = 3;
    // Prepare 5 unique vectors
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,    // vector 0
        4.0f, 5.0f, 6.0f,    // vector 1
        7.0f, 8.0f, 9.0f,    // vector 2
        10.0f, 11.0f, 12.0f, // vector 3
        13.0f, 14.0f, 15.0f  // vector 4
    };

    // Use L2 metric for this test
    std::unique_ptr<MockFaissMethods> mockFaissMethods(new MockFaissMethods());
    knn_jni::faiss_wrapper::IndexService indexService(std::move(mockFaissMethods));

    jlong indexPtr = indexService.buildFlatIndexFromVectors(numVectors, dim, data, faiss::METRIC_L2);

    ASSERT_NE(indexPtr, 0);
    auto* index = reinterpret_cast<faiss::IndexFlatL2*>(indexPtr);
    ASSERT_EQ(index->ntotal, numVectors);

    // Check each vector in the index matches input and order
    std::vector<float> reconstructed(dim);
    for (int i = 0; i < numVectors; ++i) {
        index->reconstruct(i, reconstructed.data());
        for (int j = 0; j < dim; ++j) {
            float expected = data[i * dim + j];
            ASSERT_FLOAT_EQ(reconstructed[j], expected)
                << "Vector " << i << " element " << j << " mismatch";
        }
    }

    // Clean up
    delete index;
}