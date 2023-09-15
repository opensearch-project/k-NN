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
#include "nmslib_wrapper.h"

#include <vector>
#include <malloc.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "test_util.h"
#include "faiss/utils/utils.h"

using ::testing::NiceMock;
using ::testing::Return;
#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

TEST(FaissHNSWIndexMemoryTest, BasicAssertions) {

    char dataset[] = "dataset/sift/sift_base.fvecs";
    float* data_load = NULL;
    unsigned points_num, dim;
    test_util::load_data(dataset, data_load, points_num, dim);

    GTEST_COUT << "points_num："<< points_num << " data dimension：" << dim << std::endl;
    float* dataptr = data_load;

    faiss::idx_t numIds = points_num;
    std::vector<faiss::idx_t> ids(points_num);
    std::vector<std::vector<float>> vectors;
    for (int64_t i = 0; i < numIds; ++i) {
        ids[i] = i;

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(*dataptr);
	    dataptr++;
        }
        vectors.push_back(vect);
    }

    free(data_load);
    malloc_trim(0);

    std::string indexPath = "tmp/FaissHNSWIndexMemoryTest.faiss";
    std::string spaceType = knn_jni::L2;
    std::string index_description = "HNSW32,Flat";  
    int thread_num = 7;
    //int efConstruction = 512;
    //int efSearch = 512;
    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&index_description;
    parametersMap[knn_jni::INDEX_THREAD_QUANTITY] = (jobject)&thread_num;
    //parametersMap[knn_jni::EF_CONSTRUCTION] = (jobject)&efConstruction;
    //parametersMap[knn_jni::EF_SEARCH] = (jobject)&efSearch;

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
    // std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

    // Clean up
    ids.clear();
    ids.shrink_to_fit();
    vectors.clear();
    vectors.shrink_to_fit();

    malloc_trim(0);
    size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);

    GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}

TEST(LIBHNSWIndexMemoryTest, BasicAssertions) {

    similarity::initLibrary();
    char dataset[] = "dataset/sift/sift_base.fvecs";
    float* data_load = NULL;
    unsigned points_num, dim;
    test_util::load_data(dataset, data_load, points_num, dim);

    GTEST_COUT << "points_num："<< points_num << " data dimension：" << dim << std::endl;
    float* dataptr = data_load;

    faiss::idx_t numIds = points_num;
    std::vector<int> ids(points_num);
    std::vector<std::vector<float>> vectors;
    for (int64_t i = 0; i < numIds; ++i) {
        ids[i] = i;

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(*dataptr);
	    dataptr++;
        }
        vectors.push_back(vect);
    }

    free(data_load);
    malloc_trim(0);

    std::string indexPath = "tmp/LibHNSWIndexMemoryTest.faiss";
    std::string spaceType = knn_jni::L2;
    int thread_num = 7;
    int efConstruction = 512;
    int efSearch = 512;
    int m = 32;

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_THREAD_QUANTITY] = (jobject)&thread_num;
    parametersMap[knn_jni::EF_CONSTRUCTION] = (jobject)&efConstruction;
    parametersMap[knn_jni::EF_SEARCH] = (jobject)&efSearch;
    parametersMap[knn_jni::M] = (jobject)&m;

    // Set up jni
    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    EXPECT_CALL(mockJNIUtil,
                GetJavaObjectArrayLength(
                        jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

    EXPECT_CALL(mockJNIUtil,
                GetJavaIntArrayLength(jniEnv, reinterpret_cast<jintArray>(&ids)))
            .WillRepeatedly(Return(ids.size()));
    // Create the index
    knn_jni::nmslib_wrapper::CreateIndex(
            &mockJNIUtil, jniEnv, reinterpret_cast<jintArray>(&ids),
            reinterpret_cast<jobjectArray>(&vectors), (jstring)&indexPath,
            (jobject)&parametersMap);

    // Make sure index can be loaded
    // std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

    // Clean up
    ids.clear();
    ids.shrink_to_fit();
    vectors.clear();
    vectors.shrink_to_fit();

    malloc_trim(0);
    size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);

    GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}

TEST(FaissNSGIndexMemoryTest, BasicAssertions) {

    char dataset[] = "dataset/sift/sift_base.fvecs";
    float* data_load = NULL;
    unsigned points_num, dim;
    test_util::load_data(dataset, data_load, points_num, dim);

    GTEST_COUT << "points_num："<< points_num << " data dimension：" << dim << std::endl;
    float* dataptr = data_load;

    faiss::idx_t numIds = points_num;
    std::vector<faiss::idx_t> ids(points_num);
    std::vector<std::vector<float>> vectors;
    for (int64_t i = 0; i < numIds; ++i) {
        ids[i] = i;

        std::vector<float> vect;
        vect.reserve(dim);
        for (int j = 0; j < dim; ++j) {
            vect.push_back(*dataptr);
	    dataptr++;
        }
        vectors.push_back(vect);
    }

    free(data_load);
    malloc_trim(0);

    std::string indexPath = "tmp/FaissNSGIndexMemoryTest.faiss";
    std::string spaceType = knn_jni::L2;
    std::string index_description = "NSG64,Flat";
    int thread_num = 7;
    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&index_description;
    parametersMap[knn_jni::INDEX_THREAD_QUANTITY] = (jobject)&thread_num;

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
    // std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

    // Clean up
    ids.clear();
    ids.shrink_to_fit();
    vectors.clear();
    vectors.shrink_to_fit();

    malloc_trim(0);
    size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);
    GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}

TEST(FaissNSGQueryMemoryTest, BasicAssertions) {

   std::string indexPath = "tmp/FaissNSGIndexMemoryTest.faiss";
   std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));
   float queryVector[128];
   float distance[10];
   faiss::idx_t ids[10];
   memset(queryVector, 0, sizeof(queryVector));
   test_util::FaissQueryIndex(index.get(), queryVector, 10, distance, ids );
   size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);
   GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}

TEST(FaissHNSWQueryMemoryTest, BasicAssertions) {

   std::string indexPath = "tmp/FaissHNSWIndexMemoryTest.faiss";
   std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));
   float queryVector[128];
   float distance[10];
   faiss::idx_t ids[10];
   memset(queryVector, 0, sizeof(queryVector));
   test_util::FaissQueryIndex(index.get(), queryVector, 10, distance, ids );
   size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);
   GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}

TEST(LIBHNSWQueryMemoryTest, BasicAssertions) {

    similarity::initLibrary();
    std::string indexPath = "tmp/LibHNSWIndexMemoryTest2.faiss";
    std::string spaceType = knn_jni::L2;
    int thread_num = 7;
    int efConstruction = 512;
    int efSearch = 512;
    int m = 32;

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_THREAD_QUANTITY] = (jobject)&thread_num;
    parametersMap[knn_jni::EF_CONSTRUCTION] = (jobject)&efConstruction;
    parametersMap[knn_jni::EF_SEARCH] = (jobject)&efSearch;
    parametersMap[knn_jni::M] = (jobject)&m;

    JNIEnv *jniEnv = nullptr;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    std::unique_ptr<knn_jni::nmslib_wrapper::IndexWrapper> loadedIndex(
            reinterpret_cast<knn_jni::nmslib_wrapper::IndexWrapper *>(
                    knn_jni::nmslib_wrapper::LoadIndex(&mockJNIUtil, jniEnv,
                                                       (jstring)&indexPath,
                                                       (jobject)&parametersMap)));

   float queryVector[128];
   float distance[10];
   faiss::idx_t ids[10];
   memset(queryVector, 0, sizeof(queryVector));
   size_t mem_usage = faiss::get_mem_usage_kb() / (1 << 10);
   GTEST_COUT<<"======Memory Usage:[" << mem_usage << "mb]======" << std::endl;
}
