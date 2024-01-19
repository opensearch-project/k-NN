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

#include "faiss_wrapper.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "test_util.h"

using ::testing::NiceMock;
using ::testing::Return;

TEST(FaissCreateIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<std::vector<float>> vectors;
    int dim = 2;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        vectors.push_back(vect);
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string spaceType = knn_jni::L2;
    std::string index_description = "HNSW32,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&index_description;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    EXPECT_CALL(mockJNIUtil,
                GetJavaObjectArrayLength(
                        jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

    // Create the index
    knn_jni::faiss_wrapper::CreateIndex(
            &mockJNIUtil, jniEnv, reinterpret_cast<jintArray>(&ids),
            reinterpret_cast<jobjectArray>(&vectors), (jstring)&indexPath,
            (jobject)&parametersMap);

    // Make sure index can be loaded
    std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(FaissCreateIndexFromTemplateTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<std::vector<float>> vectors;
    int dim = 2;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        vectors.push_back(vect);
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto vectorIoWriter = test_util::FaissGetSerializedIndex(createdIndex.get());

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    EXPECT_CALL(mockJNIUtil,
                GetJavaObjectArrayLength(
                        jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

    std::string spaceType = knn_jni::L2;
    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;

    knn_jni::faiss_wrapper::CreateIndexFromTemplate(
            &mockJNIUtil, jniEnv, reinterpret_cast<jintArray>(&ids),
            reinterpret_cast<jobjectArray>(&vectors), (jstring)&indexPath,
            reinterpret_cast<jbyteArray>(&(vectorIoWriter.data)),
            (jobject) &parametersMap
            );

    // Make sure index can be loaded
    std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(FaissLoadIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int dim = 2;
    for (int64_t i = 0; i < numIds; i++) {
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    test_util::FaissWriteIndex(&createdIndexWithData, indexPath);

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, jniEnv, (jstring)&indexPath)));

    // Compare serialized versions
    auto createIndexSerialization =
            test_util::FaissGetSerializedIndex(&createdIndexWithData);
    auto loadedIndexSerialization = test_util::FaissGetSerializedIndex(
            reinterpret_cast<faiss::Index *>(loadedIndexPointer.get()));

    ASSERT_NE(0, loadedIndexSerialization.data.size());
    ASSERT_EQ(createIndexSerialization.data.size(),
              loadedIndexSerialization.data.size());

    for (int i = 0; i < loadedIndexSerialization.data.size(); ++i) {
        ASSERT_EQ(createIndexSerialization.data[i],
                  loadedIndexSerialization.data[i]);
    }

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(FaissQueryIndexTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int dim = 16;
    for (int64_t i = 0; i < numIds; i++) {
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int k = 10;
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(2, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissQueryIndexWithParentFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<int> parentIds;
    int dim = 16;
    for (int64_t i = 1; i < numIds + 1; i++) {
        if (i % 10 == 0) {
            parentIds.push_back(i);
            continue;
        }
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int k = 20;
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(2, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                GetJavaIntArrayLength(
                        jniEnv, reinterpret_cast<jintArray>(&parentIds)))
            .WillRepeatedly(Return(parentIds.size()));
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k,
                                reinterpret_cast<jintArray>(&parentIds))));

        // Even with k 20, result should have only 10 which is total number of groups
        ASSERT_EQ(10, results->size());
        // Result should be one for each group
        std::set<int> idSet;
        for (const auto& pairPtr : *results) {
            idSet.insert(pairPtr->first / 10);
        }
        ASSERT_EQ(10, idSet.size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissFreeTest, BasicAssertions) {
    // Define the data
    int dim = 2;
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    faiss::Index *createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));

    // Free created index --> memory check should catch failure
    knn_jni::faiss_wrapper::Free(reinterpret_cast<jlong>(createdIndex));
}

TEST(FaissInitLibraryTest, BasicAssertions) {
    knn_jni::faiss_wrapper::InitLibrary();
}

TEST(FaissTrainIndexTest, BasicAssertions) {
    // Define the index configuration
    int dim = 2;
    std::string spaceType = knn_jni::L2;
    std::string index_description = "IVF4,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject) &index_description;

    // Define training data
    int numTrainingVectors = 256;
    std::vector<float> trainingVectors;

    for (int i = 0; i < numTrainingVectors; ++i) {
        for (int j = 0; j < dim; ++j) {
            trainingVectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Perform training
    std::unique_ptr<std::vector<uint8_t>> trainedIndexSerialization(
            reinterpret_cast<std::vector<uint8_t> *>(
                    knn_jni::faiss_wrapper::TrainIndex(
                            &mockJNIUtil, jniEnv, (jobject) &parametersMap, dim,
                            reinterpret_cast<jlong>(&trainingVectors))));

    std::unique_ptr<faiss::Index> trainedIndex(
            test_util::FaissLoadFromSerializedIndex(trainedIndexSerialization.get()));

    // Confirm that training succeeded
    ASSERT_TRUE(trainedIndex->is_trained);
}
