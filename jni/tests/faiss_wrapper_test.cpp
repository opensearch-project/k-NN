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
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFPQ.h"
#include "mocks/faiss_index_service_mock.h"
#include "native_stream_support_util.h"

using ::test_util::JavaFileIndexOutputMock;
using ::test_util::MockJNIUtil;
using ::test_util::StreamIOError;
using ::test_util::setUpJavaFileOutputMocking;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::_;

const float randomDataMin = -500.0;
const float randomDataMax = 500.0;
const float rangeSearchRandomDataMin = -50;
const float rangeSearchRandomDataMax = 50;
const float rangeSearchRadius = 20000;

void createIndexIteratively(
        knn_jni::JNIUtilInterface * JNIUtil, 
        JNIEnv *jniEnv, 
        std::vector<faiss::idx_t> & ids,
        std::vector<float> & vectors,
        int dim,
        jobject javaFileOutputMock,
        std::unordered_map<string, jobject> parametersMap,
        IndexService * indexService,
        int insertions = 10
    ) {
    long numDocs = ids.size();
    if (numDocs % insertions != 0) {
        throw std::invalid_argument("Number of documents should be divisible by number of insertions");
    }
    long docsPerInsertion = numDocs / insertions;
    long index_ptr = knn_jni::faiss_wrapper::InitIndex(JNIUtil, jniEnv, numDocs, dim, (jobject)&parametersMap, indexService);
    std::vector<faiss::idx_t> insertIds;
    std::vector<float> insertVecs;
    for (int i = 0; i < insertions; i++) {
        insertIds.clear();
        insertVecs.clear();
        int start_idx = i * docsPerInsertion;
        int end_idx = start_idx + docsPerInsertion;
        for (int j = start_idx; j < end_idx; j++) {
            insertIds.push_back(j);
            for(int k = 0; k < dim; k++) {
                insertVecs.push_back(vectors[j * dim + k]);
            }
        }
        knn_jni::faiss_wrapper::InsertToIndex(JNIUtil, jniEnv, reinterpret_cast<jintArray>(&insertIds), (jlong)&insertVecs, dim, index_ptr, 0, indexService);
    }
    knn_jni::faiss_wrapper::WriteIndex(JNIUtil, jniEnv, javaFileOutputMock, index_ptr, indexService);
}

void createBinaryIndexIteratively(
        knn_jni::JNIUtilInterface * JNIUtil, 
        JNIEnv *jniEnv, 
        std::vector<faiss::idx_t> & ids,
        std::vector<uint8_t> & vectors,
        int dim,
        jobject javaFileOutputMock,
        std::unordered_map<string, jobject> parametersMap, 
        IndexService * indexService,
        int insertions = 10
    ) {
    long numDocs = ids.size();
    long index_ptr = knn_jni::faiss_wrapper::InitIndex(JNIUtil, jniEnv, numDocs, dim, (jobject)&parametersMap, indexService);
    std::vector<faiss::idx_t> insertIds;
    std::vector<float> insertVecs;
    for (int i = 0; i < insertions; i++) {
        int start_idx = numDocs * i / insertions;
        int end_idx = numDocs * (i + 1) / insertions;
        int docs_to_insert = end_idx - start_idx;
        if (docs_to_insert == 0) {
            continue;
        }
        insertIds.clear();
        insertVecs.clear();
        for (int j = start_idx; j < end_idx; j++) {
            insertIds.push_back(j);
            for(int k = 0; k < dim / 8; k++) {
                insertVecs.push_back(vectors[j * (dim / 8) + k]);
            }
        }
        knn_jni::faiss_wrapper::InsertToIndex(JNIUtil, jniEnv, reinterpret_cast<jintArray>(&insertIds), (jlong)&insertVecs, dim, index_ptr, 0, indexService);
    }

    knn_jni::faiss_wrapper::WriteIndex(JNIUtil, jniEnv, javaFileOutputMock, index_ptr, indexService);
}

TEST(FaissCreateIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
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
    std::string spaceType = knn_jni::L2;
    std::string indexDescription = "HNSW32,Flat";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&indexDescription;
    std::unordered_map<std::string, jobject> subParametersMap;
    parametersMap[knn_jni::PARAMETERS] = (jobject)&subParametersMap;

    // Set up jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
    setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, false);

    // Create the index
    std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
    NiceMock<MockIndexService> mockIndexService(std::move(faissMethods));
    int insertions = 10;
    EXPECT_CALL(mockIndexService, initIndex(_, _, faiss::METRIC_L2, indexDescription, dim, (int)numIds, 0, subParametersMap))
        .Times(1);
    EXPECT_CALL(mockIndexService, insertToIndex(dim, numIds / insertions, 0, _, _, _))
        .Times(insertions);
    EXPECT_CALL(mockIndexService, writeIndex(_, _))
        .Times(1);

    createIndexIteratively(&mockJNIUtil,
                           &jniEnv,
                           ids,
                           vectors,
                           dim,
                           (jobject) (&javaFileIndexOutputMock),
                           parametersMap,
                           &mockIndexService,
                           insertions);
}

TEST(FaissCreateBinaryIndexTest, BasicAssertions) {
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
    std::string spaceType = knn_jni::HAMMING;
    std::string indexDescription = "BHNSW32";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&indexDescription;
    std::unordered_map<std::string, jobject> subParametersMap;
    parametersMap[knn_jni::PARAMETERS] = (jobject)&subParametersMap;

    // Set up jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
    setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, false);

    // Create the index
    std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
    NiceMock<MockIndexService> mockIndexService(std::move(faissMethods));
    int insertions = 10;
    EXPECT_CALL(mockIndexService, initIndex(_, _, faiss::METRIC_L2, indexDescription, dim, (int)numIds, 0, subParametersMap))
        .Times(1);
    EXPECT_CALL(mockIndexService, insertToIndex(dim, numIds / insertions, 0, _, _, _))
        .Times(insertions);
    EXPECT_CALL(mockIndexService, writeIndex(_, _))
        .Times(1);

    // This method calls delete vectors at the end
    createBinaryIndexIteratively(&mockJNIUtil,
                                 &jniEnv,
                                 ids,
                                 vectors,
                                 dim,
                                 (jobject) (&javaFileIndexOutputMock),
                                 parametersMap,
                                 &mockIndexService,
                                 insertions);
}

TEST(FaissCreateIndexFromTemplateTest, BasicAssertions) {
    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        // Define the data
        faiss::idx_t numIds = 100;
        std::vector<faiss::idx_t> ids;
        auto *vectors = new std::vector<float>();
        int dim = 2;
        vectors->reserve(dim * numIds);
        for (int64_t i = 0; i < numIds; ++i) {
          ids.push_back(i);
          for (int j = 0; j < dim; ++j) {
            vectors->push_back(test_util::RandomFloat(-500.0, 500.0));
          }
        }

        std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
        faiss::MetricType metricType = faiss::METRIC_L2;
        std::string method = "HNSW32,Flat";

        std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
        auto vectorIoWriter = test_util::FaissGetSerializedIndex(createdIndex.get());

        // Setup jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        std::string spaceType = knn_jni::L2;
        std::unordered_map<std::string, jobject> parametersMap;
        parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;

        try {
            knn_jni::faiss_wrapper::CreateIndexFromTemplate(
                &mockJNIUtil, &jniEnv, reinterpret_cast<jintArray>(&ids),
                (jlong)vectors, dim, (jobject)(&javaFileIndexOutputMock),
                reinterpret_cast<jbyteArray>(&(vectorIoWriter.data)),
                (jobject) &parametersMap);
            javaFileIndexOutputMock.file_writer.close();
        } catch (const StreamIOError& e) {
            ASSERT_TRUE(throwIOException);
            continue;
        }

        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissCreateByteIndexFromTemplateTest, BasicAssertions) {
    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        // Define the data
        faiss::idx_t numIds = 100;
        std::vector<faiss::idx_t> ids;
        auto *vectors = new std::vector<int8_t>();
        int dim = 8;
        vectors->reserve(dim * numIds);
        for (int64_t i = 0; i < numIds; ++i) {
          ids.push_back(i);
          for (int j = 0; j < dim; ++j) {
            vectors->push_back(test_util::RandomInt(-128, 127));
          }
        }

        std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
        faiss::MetricType metricType = faiss::METRIC_L2;
        std::string method = "HNSW32,SQ8_direct_signed";

        std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
        auto vectorIoWriter = test_util::FaissGetSerializedIndex(createdIndex.get());

        // Setup jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        std::string spaceType = knn_jni::L2;
        std::unordered_map<std::string, jobject> parametersMap;
        parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;

        try {
            knn_jni::faiss_wrapper::CreateByteIndexFromTemplate(
                &mockJNIUtil, &jniEnv, reinterpret_cast<jintArray>(&ids),
                (jlong) vectors, dim, (jstring) (&javaFileIndexOutputMock),
                reinterpret_cast<jbyteArray>(&(vectorIoWriter.data)),
                (jobject) &parametersMap
            );

            // Make sure we close a file stream before reopening the created file.
            javaFileIndexOutputMock.file_writer.close();
        } catch (const StreamIOError& e) {
            ASSERT_TRUE(throwIOException);
            continue;
        }

        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissLoadIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 100;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

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
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

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

TEST(FaissLoadBinaryIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    auto vectors = std::vector<uint8_t>(numIds);
    int dim = 128;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    std::string method = "BHNSW32";

    // Create the index
    std::unique_ptr<faiss::IndexBinary> createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
            test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    test_util::FaissWriteBinaryIndex(&createdIndexWithData, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::IndexBinary> loadedIndexPointer(
            reinterpret_cast<faiss::IndexBinary *>(knn_jni::faiss_wrapper::LoadBinaryIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Compare serialized versions
    auto createIndexSerialization =
            test_util::FaissGetSerializedBinaryIndex(&createdIndexWithData);
    auto loadedIndexSerialization = test_util::FaissGetSerializedBinaryIndex(
            reinterpret_cast<faiss::IndexBinary *>(loadedIndexPointer.get()));

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

TEST(FaissLoadIndexTest, HNSWPQDisableSdcTable) {
    // Check that when we load an HNSWPQ index, the sdc table is not present.
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "HNSW16,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Cast down until we get to the pq backed storage index and checke the size of the table
    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto hnswPQIndex = dynamic_cast<faiss::IndexHNSWPQ *>(idMapIndex->index);
    ASSERT_NE(hnswPQIndex, nullptr);
    auto pqIndex = dynamic_cast<faiss::IndexPQ*>(hnswPQIndex->storage);
    ASSERT_NE(pqIndex, nullptr);
    ASSERT_EQ(0, pqIndex->pq.sdc_table.size());
}

TEST(FaissLoadIndexTest, IVFPQDisablePrecomputeTable) {
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "IVF4,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    // Cast down until we get to the ivfpq-l2 state
    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto ivfpqIndex = dynamic_cast<faiss::IndexIVFPQ *>(idMapIndex->index);
    ASSERT_NE(ivfpqIndex, nullptr);
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
}

TEST(FaissQueryIndexTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    int dim = 16;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int k = 10;
    int efSearch = 20;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

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
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    auto methodParamsJ = reinterpret_cast<jobject>(&methodParams);

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, methodParamsJ, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

TEST(FaissQueryBinaryIndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    auto vectors = std::vector<uint8_t>(numIds);
    int dim = 128;
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    // Define query data
    int k = 10;
    int numQueries = 100;
    std::vector<std::vector<uint8_t>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<uint8_t> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomInt(0, 255));
        }
        queries.push_back(query);
    }

    // Create the index
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
            test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, int32_t> *>> results(
                reinterpret_cast<std::vector<std::pair<int, int32_t> *> *>(
                        knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jbyteArray>(&query), k, nullptr, nullptr, 0, nullptr)));

        ASSERT_EQ(k, results->size());

        // Need to free up each result
        for (auto it : *results.get()) {
            delete it;
        }
    }
}

//Test for a bug reported in https://github.com/opensearch-project/k-NN/issues/1435
TEST(FaissQueryIndexWithFilterTest1435, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<std::vector<float>> queries;

    int dim = 16;
    for (int64_t i = 1; i < numIds + 1; i++) {
        std::vector<float> query;
        query.reserve(dim);
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            float vector = test_util::RandomFloat(-500.0, 500.0);
            vectors.push_back(vector);
            query.push_back(vector);
        }
        queries.push_back(query);
    }

    int num_bits = test_util::bits2words(164);
    std::vector<jlong> bitmap(num_bits,0);
    std::vector<int64_t> filterIds;

    for (int64_t i = 154; i < 163; i++) {
        filterIds.push_back(i);
        test_util::setBitSet(i, bitmap.data(), bitmap.size());
    }
    std::unordered_set<int> filterIdSet(filterIds.begin(), filterIds.end());

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                GetJavaLongArrayLength(
                        &jniEnv, reinterpret_cast<jlongArray>(&bitmap)))
            .WillRepeatedly(Return(bitmap.size()));

    int k = 20;
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex_WithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, nullptr,
                                reinterpret_cast<jlongArray>(&bitmap), 0, nullptr)));

        ASSERT_TRUE(results->size() <= filterIds.size());
        ASSERT_TRUE(results->size() > 0);
        for (const auto& pairPtr : *results) {
            auto it = filterIdSet.find(pairPtr->first);
            ASSERT_NE(it, filterIdSet.end());
        }

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
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    int efSearch = 100;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                    GetJavaIntArrayLength(
                            &jniEnv, reinterpret_cast<jintArray>(&parentIds)))
                .WillRepeatedly(Return(parentIds.size()));
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, reinterpret_cast<jobject>(&methodParams),
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

TEST(FaissQueryIndexHNSWCagraWithParentFilterTest, BasicAssertions) {
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
    std::string method = "HNSW32,Cagra";

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
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);
    dynamic_cast<faiss::IndexHNSWCagra*>(createdIndexWithData.index)->base_level_only=true;

    int efSearch = 100;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                    GetJavaIntArrayLength(
                            &jniEnv, reinterpret_cast<jintArray>(&parentIds)))
                .WillRepeatedly(Return(parentIds.size()));
    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(
                        knn_jni::faiss_wrapper::QueryIndex(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), k, reinterpret_cast<jobject>(&methodParams),
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
    knn_jni::faiss_wrapper::Free(reinterpret_cast<jlong>(createdIndex), JNI_FALSE);
}


TEST(FaissBinaryFreeTest, BasicAssertions) {
    // Define the data
    int dim = 8;
    std::string method = "BHNSW32";

    // Create the index
    faiss::IndexBinary *createdIndex(
            test_util::FaissCreateBinaryIndex(dim, method));

    // Free created index --> memory check should catch failure
    knn_jni::faiss_wrapper::Free(reinterpret_cast<jlong>(createdIndex), JNI_TRUE);
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
    std::vector<float> trainingVectors = test_util::RandomVectors(dim, numTrainingVectors, randomDataMin, randomDataMax);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Perform training
    std::unique_ptr<std::vector<uint8_t>> trainedIndexSerialization(
            reinterpret_cast<std::vector<uint8_t> *>(
                    knn_jni::faiss_wrapper::TrainIndex(
                            &mockJNIUtil, &jniEnv, (jobject) &parametersMap, dim,
                            reinterpret_cast<jlong>(&trainingVectors))));

    std::unique_ptr<faiss::Index> trainedIndex(
            test_util::FaissLoadFromSerializedIndex(trainedIndexSerialization.get()));

    // Confirm that training succeeded
    ASSERT_TRUE(trainedIndex->is_trained);
}

TEST(FaissTrainByteIndexTest, BasicAssertions) {
    // Define the index configuration
    int dim = 2;
    std::string spaceType = knn_jni::L2;
    std::string index_description = "IVF4,SQ8_direct_signed";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject) &spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject) &index_description;

    // Define training data
    int numTrainingVectors = 256;
    std::vector<int8_t> trainingVectors = test_util::RandomByteVectors(dim, numTrainingVectors, -128, 127);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    // Perform training
    std::unique_ptr<std::vector<uint8_t>> trainedIndexSerialization(
            reinterpret_cast<std::vector<uint8_t> *>(
                    knn_jni::faiss_wrapper::TrainByteIndex(
                            &mockJNIUtil, &jniEnv, (jobject) &parametersMap, dim,
                            reinterpret_cast<jlong>(&trainingVectors))));

    std::unique_ptr<faiss::Index> trainedIndex(
            test_util::FaissLoadFromSerializedIndex(trainedIndexSerialization.get()));

    // Confirm that training succeeded
    ASSERT_TRUE(trainedIndex->is_trained);
}

TEST(FaissCreateHnswSQfp16IndexTest, BasicAssertions) {
    // Define the data
    faiss::idx_t numIds = 200;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int dim = 2;
    vectors.reserve(dim * numIds);
    for (int64_t i = 0; i < numIds; ++i) {
        ids.push_back(i);
        for (int j = 0; j < dim; ++j) {
            vectors.push_back(test_util::RandomFloat(-500.0, 500.0));
        }
    }

    std::string spaceType = knn_jni::L2;
    std::string index_description = "HNSW32,SQfp16";

    std::unordered_map<std::string, jobject> parametersMap;
    parametersMap[knn_jni::SPACE_TYPE] = (jobject)&spaceType;
    parametersMap[knn_jni::INDEX_DESCRIPTION] = (jobject)&index_description;

    for (auto throwIOException : std::array<bool, 2> {false, true}) {
        const std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");

        // Set up jni
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        JavaFileIndexOutputMock javaFileIndexOutputMock {indexPath};
        setUpJavaFileOutputMocking(javaFileIndexOutputMock, mockJNIUtil, throwIOException);

        EXPECT_CALL(mockJNIUtil,
                    GetJavaObjectArrayLength(
                        &jniEnv, reinterpret_cast<jobjectArray>(&vectors)))
            .WillRepeatedly(Return(vectors.size()));

        // Create the index
        std::unique_ptr<FaissMethods> faissMethods(new FaissMethods());
        knn_jni::faiss_wrapper::IndexService IndexService(std::move(faissMethods));

        try {
            createIndexIteratively(&mockJNIUtil, &jniEnv, ids, vectors, dim, (jobject) (&javaFileIndexOutputMock), parametersMap, &IndexService);
            // Make sure we close a file stream before reopening the created file.
            javaFileIndexOutputMock.file_writer.close();
        } catch (const std::exception& e) {
            ASSERT_STREQ("Failed to write index to disk", e.what());
            ASSERT_TRUE(throwIOException);
            continue;
        }
        ASSERT_FALSE(throwIOException);

        // Make sure index can be loaded
        std::unique_ptr<faiss::Index> index(test_util::FaissLoadIndex(indexPath));
        auto indexIDMap =  dynamic_cast<faiss::IndexIDMap*>(index.get());

        // Assert that Index is of type IndexHNSWSQ
        ASSERT_NE(indexIDMap, nullptr);
        ASSERT_NE(dynamic_cast<faiss::IndexHNSWSQ*>(indexIDMap->index), nullptr);

        // Clean up
        std::remove(indexPath.c_str());
    }  // End for
}

TEST(FaissIsSharedIndexStateRequired, BasicAssertions) {
    int d = 128;
    int hnswM = 16;
    int ivfNlist = 4;
    int pqM = 1;
    int pqCodeSize = 8;
    std::unique_ptr<faiss::IndexHNSW> indexHNSWL2(new faiss::IndexHNSW(d, hnswM, faiss::METRIC_L2));
    std::unique_ptr<faiss::IndexIVFPQ> indexIVFPQIP(new faiss::IndexIVFPQ(
                new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT),
                d,
                ivfNlist,
                pqM,
                pqCodeSize,
                faiss::METRIC_INNER_PRODUCT
            ));
    std::unique_ptr<faiss::IndexIVFPQ> indexIVFPQL2(new faiss::IndexIVFPQ(
                new faiss::IndexFlat(d, faiss::METRIC_L2),
                d,
                ivfNlist,
                pqM,
                pqCodeSize,
                faiss::METRIC_L2
            ));
    std::unique_ptr<faiss::IndexIDMap> indexIDMapIVFPQL2(new faiss::IndexIDMap(
                new faiss::IndexIVFPQ(
                        new faiss::IndexFlat(d, faiss::METRIC_L2),
                        d,
                        ivfNlist,
                        pqM,
                        pqCodeSize,
                        faiss::METRIC_L2
                )
            ));
    std::unique_ptr<faiss::IndexIDMap> indexIDMapIVFPQIP(new faiss::IndexIDMap(
                new faiss::IndexIVFPQ(
                        new faiss::IndexFlat(d, faiss::METRIC_INNER_PRODUCT),
                        d,
                        ivfNlist,
                        pqM,
                        pqCodeSize,
                        faiss::METRIC_INNER_PRODUCT
                )
            ));
    jlong nullAddress = 0;

    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexHNSWL2.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIVFPQIP.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIDMapIVFPQIP.get()));
    ASSERT_FALSE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) nullAddress));

    ASSERT_TRUE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIVFPQL2.get()));
    ASSERT_TRUE(knn_jni::faiss_wrapper::IsSharedIndexStateRequired((jlong) indexIDMapIVFPQL2.get()));
}

TEST(FaissInitAndSetSharedIndexState, BasicAssertions) {
    faiss::idx_t numIds = 256;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, randomDataMin, randomDataMax);

    std::string indexPath = test_util::RandomString(10, "tmp/", ".faiss");
    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string indexDescription = "IVF4,PQ1x4";

    std::unique_ptr<faiss::Index> faissIndex(test_util::FaissCreateIndex(dim, indexDescription, metricType));
    test_util::FaissTrainIndex(faissIndex.get(), numIds, vectors.data());
    auto faissIndexWithIDMap = test_util::FaissAddData(faissIndex.get(), ids, vectors);
    test_util::FaissWriteIndex(&faissIndexWithIDMap, indexPath);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    std::unique_ptr<faiss::Index> loadedIndexPointer(
            reinterpret_cast<faiss::Index *>(knn_jni::faiss_wrapper::LoadIndex(
                    &mockJNIUtil, &jniEnv, (jstring)&indexPath)));

    auto idMapIndex = dynamic_cast<faiss::IndexIDMap *>(loadedIndexPointer.get());
    ASSERT_NE(idMapIndex, nullptr);
    auto ivfpqIndex = dynamic_cast<faiss::IndexIVFPQ *>(idMapIndex->index);
    ASSERT_NE(ivfpqIndex, nullptr);
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
    jlong sharedModelAddress = knn_jni::faiss_wrapper::InitSharedIndexState((jlong) loadedIndexPointer.get());
    ASSERT_EQ(0, ivfpqIndex->precomputed_table->size());
    knn_jni::faiss_wrapper::SetSharedIndexState((jlong) loadedIndexPointer.get(), sharedModelAddress);
    ASSERT_EQ(sharedModelAddress, (jlong) ivfpqIndex->precomputed_table);
    ASSERT_NE(0, ivfpqIndex->precomputed_table->size());
    ASSERT_EQ(1, ivfpqIndex->use_precomputed_table);
    knn_jni::faiss_wrapper::FreeSharedIndexState(sharedModelAddress);
}

TEST(FaissRangeSearchQueryIndexTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    int efSearch = 20;
    std::unordered_map<std::string, jobject> methodParams;
    methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
    auto methodParamsJ = reinterpret_cast<jobject>(&methodParams);

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int maxResultWindow = 20000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearch(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, methodParamsJ, maxResultWindow, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());


        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTest_WhenHitMaxWindowResult, BasicAssertions){
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int maxResultWindow = 10;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearch(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        // assert result size is equal to maxResultWindow
        ASSERT_EQ(maxResultWindow, results->size());

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTestWithFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 200;
    int dim = 2;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<float> vectors = test_util::RandomVectors(dim, numIds, rangeSearchRandomDataMin, rangeSearchRandomDataMax);

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 100;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;

    int num_bits = test_util::bits2words(164);
    std::vector<jlong> bitmap(num_bits,0);
    std::vector<int64_t> filterIds;

    for (int64_t i = 1; i < 50; i++) {
        filterIds.push_back(i);
        test_util::setBitSet(i, bitmap.data(), bitmap.size());
    }
    std::unordered_set<int> filterIdSet(filterIds.begin(), filterIds.end());

    int maxResultWindow = 20000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearchWithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow,
                                reinterpret_cast<jlongArray>(&bitmap), 0, nullptr)));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        ASSERT_TRUE(results->size() <= filterIds.size());
        for (const auto& pairPtr : *results) {
            auto it = filterIdSet.find(pairPtr->first);
            ASSERT_NE(it, filterIdSet.end());
        }

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}

TEST(FaissRangeSearchQueryIndexTestWithParentFilterTest, BasicAssertions) {
    // Define the index data
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    std::vector<int> parentIds;
    int dim = 2;
    for (int64_t i = 1; i < numIds + 1; i++) {
        if (i % 10 == 0) {
            parentIds.push_back(i);
            continue;
        }
        ids.push_back(i);
        for (int j = 0; j < dim; j++) {
            vectors.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
    }

    faiss::MetricType metricType = faiss::METRIC_L2;
    std::string method = "HNSW32,Flat";

    // Define query data
    int numQueries = 1;
    std::vector<std::vector<float>> queries;

    for (int i = 0; i < numQueries; i++) {
        std::vector<float> query;
        query.reserve(dim);
        for (int j = 0; j < dim; j++) {
            query.push_back(test_util::RandomFloat(rangeSearchRandomDataMin, rangeSearchRandomDataMax));
        }
        queries.push_back(query);
    }

    // Create the index
    std::unique_ptr<faiss::Index> createdIndex(
            test_util::FaissCreateIndex(dim, method, metricType));
    auto createdIndexWithData =
            test_util::FaissAddData(createdIndex.get(), ids, vectors);

    // Setup jni
    NiceMock<JNIEnv> jniEnv;
    NiceMock<test_util::MockJNIUtil> mockJNIUtil;
    EXPECT_CALL(mockJNIUtil,
                GetJavaIntArrayLength(
                        &jniEnv, reinterpret_cast<jintArray>(&parentIds)))
            .WillRepeatedly(Return(parentIds.size()));

    int maxResultWindow = 10000;

    for (auto query : queries) {
        std::unique_ptr<std::vector<std::pair<int, float> *>> results(
                reinterpret_cast<std::vector<std::pair<int, float> *> *>(

                        knn_jni::faiss_wrapper::RangeSearchWithFilter(
                                &mockJNIUtil, &jniEnv,
                                reinterpret_cast<jlong>(&createdIndexWithData),
                                reinterpret_cast<jfloatArray>(&query), rangeSearchRadius, nullptr, maxResultWindow, nullptr, 0,
                                reinterpret_cast<jintArray>(&parentIds))));

        // assert result size is not 0
        ASSERT_NE(0, results->size());
        // Result should be one for each group
        std::set<int> idSet;
        for (const auto& pairPtr : *results) {
            idSet.insert(pairPtr->first / 10);
        }
        ASSERT_NE(0, idSet.size());

        // Need to free up each result
        for (auto it : *results) {
            delete it;
        }
    }
}
TEST(FaissLoadIndexWithStreamADCTest, HandlesCorruptedBinaryIndex) {
    // Create invalid/corrupted data
    std::vector<uint8_t> corruptedData = {0x00, 0x01, 0x02, 0x03, 0x04};

    // Create VectorIOReader with corrupted data
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = corruptedData;

    faiss::MetricType metricType = faiss::METRIC_L2;

    // Should throw exception for corrupted data
    EXPECT_THROW({
        knn_jni::faiss_wrapper::LoadIndexWithStreamADC(
            &vectorIoReader, metricType);
    }, std::exception);
}

TEST(FaissLoadIndexWithStreamADCTest, ValidBinaryIndexTransformation) {
    // Create a test binary index structure
    int dim = 128;
    faiss::idx_t numIds = 100;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<uint8_t> vectors;
    vectors.reserve(numIds * (dim / 8));

    for (int64_t i = 0; i < numIds; ++i) {
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    // Create binary HNSW index
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> createdIndex(
        test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
        test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    // Serialize the index
    auto serializedIndex = test_util::FaissGetSerializedBinaryIndex(&createdIndexWithData);

    // Create VectorIOReader from serialized data
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = serializedIndex.data;

    faiss::MetricType metricType = faiss::METRIC_L2;

    // Test the transformation
    jlong resultPtr = 0;
    EXPECT_NO_THROW({
        resultPtr = knn_jni::faiss_wrapper::LoadIndexWithStreamADC(
            &vectorIoReader, metricType);
    });

    ASSERT_NE(0, resultPtr);

    // Verify the result is a valid IndexIDMap
    auto* resultIndex = reinterpret_cast<faiss::IndexIDMap*>(resultPtr);
    ASSERT_NE(resultIndex, nullptr);
    ASSERT_NE(resultIndex->index, nullptr);

    ASSERT_EQ(dim, resultIndex->d);
    ASSERT_EQ(numIds, resultIndex->ntotal);

    // Verify it's an HNSW index
    auto* hnswIndex = dynamic_cast<faiss::IndexHNSW*>(resultIndex->index);
    ASSERT_NE(hnswIndex, nullptr);

    // Clean up
    knn_jni::faiss_wrapper::Free(resultPtr, JNI_FALSE);
}

TEST(FaissLoadIndexWithStreamADCTest, ValidInnerProductMetric) {
    // Create a test binary index structure
    int dim = 64;
    faiss::idx_t numIds = 50;
    std::vector<faiss::idx_t> ids = test_util::Range(numIds);
    std::vector<uint8_t> vectors;
    vectors.reserve(numIds * (dim / 8));

    for (int64_t i = 0; i < numIds; ++i) {
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    // Create binary HNSW index
    std::string method = "BHNSW16";
    std::unique_ptr<faiss::IndexBinary> createdIndex(
        test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
        test_util::FaissAddBinaryData(createdIndex.get(), ids, vectors);

    // Serialize the index
    auto serializedIndex = test_util::FaissGetSerializedBinaryIndex(&createdIndexWithData);

    // Create VectorIOReader from serialized data
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = serializedIndex.data;

    faiss::MetricType metricType = faiss::METRIC_INNER_PRODUCT;

    // Test with inner product metric
    jlong resultPtr = 0;
    EXPECT_NO_THROW({
        resultPtr = knn_jni::faiss_wrapper::LoadIndexWithStreamADC(
            &vectorIoReader, metricType);
    });

    ASSERT_NE(0, resultPtr);

    // Verify the result
    auto* resultIndex = reinterpret_cast<faiss::IndexIDMap*>(resultPtr);
    ASSERT_NE(resultIndex, nullptr);
    ASSERT_EQ(dim, resultIndex->d);
    ASSERT_EQ(numIds, resultIndex->ntotal);

    // Clean up
    knn_jni::faiss_wrapper::Free(resultPtr, JNI_FALSE);
}

TEST(FaissLoadIndexWithStreamADCTest, PreservesIdMapping) {
     // Create a test binary index with specific IDs
    int dim = 128;
    faiss::idx_t numIds = 10;
    std::vector<faiss::idx_t> customIds;
    std::vector<uint8_t> vectors;
    vectors.reserve(numIds * (dim / 8));

    // Use non-sequential IDs to test mapping preservation
    for (int64_t i = 0; i < numIds; ++i) {
        customIds.push_back(i * 10 + 100); // IDs: 100, 110, 120, ...
        for (int j = 0; j < dim / 8; ++j) {
            vectors.push_back(test_util::RandomInt(0, 255));
        }
    }

    // Create binary HNSW index
    std::string method = "BHNSW32";
    std::unique_ptr<faiss::IndexBinary> createdIndex(
        test_util::FaissCreateBinaryIndex(dim, method));
    auto createdIndexWithData =
        test_util::FaissAddBinaryData(createdIndex.get(), customIds, vectors);

    // Serialize the index
    auto serializedIndex = test_util::FaissGetSerializedBinaryIndex(&createdIndexWithData);

    // Create VectorIOReader from serialized data
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = serializedIndex.data;

    faiss::MetricType metricType = faiss::METRIC_L2;

    // Transform the index
    jlong resultPtr = knn_jni::faiss_wrapper::LoadIndexWithStreamADC(
        &vectorIoReader, metricType);

    auto* resultIndex = reinterpret_cast<faiss::IndexIDMap*>(resultPtr);
    ASSERT_NE(resultIndex, nullptr);

    // Verify ID mapping is preserved
    ASSERT_EQ(numIds, resultIndex->id_map.size());
    for (size_t i = 0; i < customIds.size(); ++i) {
        ASSERT_EQ(customIds[i], resultIndex->id_map[i]);
    }

    // Clean up
    knn_jni::faiss_wrapper::Free(resultPtr, JNI_FALSE);
}