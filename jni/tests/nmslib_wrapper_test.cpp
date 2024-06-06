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

#include "nmslib_wrapper.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "test_util.h"

using ::testing::NiceMock;
using ::testing::Return;

TEST(NmslibIndexWrapperSearchTest, BasicAssertions) {
    similarity::initLibrary();
    knn_jni::nmslib_wrapper::IndexWrapper indexWrapper = knn_jni::nmslib_wrapper::IndexWrapper("l2");
    int k = 10;
    int dim = 2;
    std::unique_ptr<float> rawQueryvector(new float[dim]);
    std::unique_ptr<similarity::Object> queryObject(new similarity::Object(-1, -1, dim*sizeof(float), rawQueryvector.get()));
    similarity::KNNQuery<float> knnQuery(*(indexWrapper.space), queryObject.get(), k);
    indexWrapper.index->Search((similarity::KNNQuery<float> *)nullptr);
}

TEST(NmslibCreateIndexTest, BasicAssertions) {
    // Initialize nmslib
    similarity::initLibrary();

    // Define index data
    int numIds = 100;
    std::vector<int> ids;
    auto *vectors = new std::vector<float>();
    int dim = 2;
    vectors->reserve(dim * numIds);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
            vectors->push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".nmslib");
    std::string spaceType = knn_jni::L2;

    std::unordered_map<std::string, jobject> parametersMap;
    int efConstruction = 512;
    int m = 96;

    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::EF_CONSTRUCTION] = (jobject)&efConstruction;
    parametersMap[knn_jni::M] = (jobject)&m;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    EXPECT_CALL(mockJNIUtil,
                GetJavaObjectArrayLength(
                        jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors->size()));

    EXPECT_CALL(mockJNIUtil,
                GetJavaIntArrayLength(jniEnv, reinterpret_cast<jintArray>(&ids)))
            .WillRepeatedly(Return(ids.size()));

    // Create the index
    knn_jni::nmslib_wrapper::CreateIndex(
            &mockJNIUtil, jniEnv, reinterpret_cast<jintArray>(&ids),
            (jlong) vectors, dim, (jstring)&indexPath,
            (jobject)&parametersMap);

    // Make sure index can be loaded
    std::unique_ptr<similarity::Space<float>> space(
            similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    spaceType, similarity::AnyParams()));
    std::vector<std::string> params;
    std::unique_ptr<similarity::Index<float>> loadedIndex(
            test_util::NmslibLoadIndex(indexPath, space.get(), spaceType, params));

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(NmslibLoadIndexTest, BasicAssertions) {
    // Initialize nmslib
    similarity::initLibrary();

    // Define index data
    int numIds = 100;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    int dim = 2;
    for (int i = 0; i < numIds; ++i) {
        ids.push_back(i);

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        vectors.push_back(vect);
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".nmslib");
    std::string spaceType = knn_jni::L2;
    std::unique_ptr<similarity::Space<float>> space(
            similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    spaceType, similarity::AnyParams()));

    std::vector<std::string> indexParameters;

    // Create index and write to disk
    std::unique_ptr<similarity::Index<float>> createdIndex(
            test_util::NmslibCreateIndex(ids.data(), vectors, space.get(), spaceType,
                                         indexParameters));

    test_util::NmslibWriteIndex(createdIndex.get(), indexPath);
    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Load index
    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;

    std::unique_ptr<knn_jni::nmslib_wrapper::IndexWrapper> loadedIndex(
            reinterpret_cast<knn_jni::nmslib_wrapper::IndexWrapper *>(
                    knn_jni::nmslib_wrapper::LoadIndex(&mockJNIUtil, jniEnv,
                                                       (jstring)&indexPath,
                                                       (jobject)&parametersMap)));

    // Check that load succeeds
    ASSERT_EQ(createdIndex->StrDesc(), loadedIndex->index->StrDesc());

    // Clean up
    std::remove(indexPath.c_str());
}

TEST(NmslibQueryIndexTest, BasicAssertions) {
    // Initialize nmslib
    similarity::initLibrary();

    // Define index data
    int numIds = 100;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    int dim = 2;
    for (int i = 0; i < numIds; ++i) {
        ids.push_back(i);

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        vectors.push_back(vect);
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".nmslib");
    std::string spaceType = knn_jni::L2;
    std::unique_ptr<similarity::Space<float>> space(
            similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    spaceType, similarity::AnyParams()));

    std::vector<std::string> indexParameters;

    // Create index
    std::unique_ptr<knn_jni::nmslib_wrapper::IndexWrapper> indexWrapper(
            new knn_jni::nmslib_wrapper::IndexWrapper(spaceType));
    indexWrapper->index.reset(test_util::NmslibCreateIndex(
            ids.data(), vectors, space.get(), spaceType, indexParameters));

    // Define query data
    int k = 10;
    int efSearch = 20;
    int numQueries = 100;
    std::vector<std::vector<float>> queries;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        queries.push_back(query);
    }

    // Setup jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Run queries
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::nmslib_wrapper::QueryIndex(
                                &mockJNIUtil, jniEnv,
                                reinterpret_cast<jlong>(indexWrapper.get()),
                                reinterpret_cast<jfloatArray>(&query), k, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto &it : *results) {
            delete it;
        }
    }
}

TEST(NmslibFreeTest, BasicAssertions) {
    // Initialize nmslib
    similarity::initLibrary();

    // Define index data
    int numIds = 100;
    std::vector<int> ids;
    std::vector<std::vector<float>> vectors;
    int dim = 2;
    for (int i = 0; i < numIds; ++i) {
        ids.push_back(i);

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
        vectors.push_back(vect);
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".nmslib");
    std::string spaceType = knn_jni::L2;
    std::unique_ptr<similarity::Space<float>> space(
            similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    spaceType, similarity::AnyParams()));

    std::vector<std::string> indexParameters;

    // Create index
    similarity::Index<float> *createdIndex = test_util::NmslibCreateIndex(
            ids.data(), vectors, space.get(), spaceType, indexParameters);

    auto *indexWrapper = new knn_jni::nmslib_wrapper::IndexWrapper(spaceType);
    indexWrapper->index.reset(createdIndex);

    // Free index
    knn_jni::nmslib_wrapper::Free(reinterpret_cast<jlong>(indexWrapper));
}

TEST(NmslibInitLibraryTest, BasicAssertions) {
    knn_jni::nmslib_wrapper::InitLibrary();
}
