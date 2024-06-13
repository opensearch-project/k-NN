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
#include "jni.h"
#include "test_util.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_factory.h"
#include "faiss/IndexIVFPQ.h"

using ::testing::NiceMock;

using idx_t = faiss::idx_t;

struct MockIndex : faiss::IndexHNSW {
    explicit MockIndex(idx_t d) : faiss::IndexHNSW(d, 32) {
    }
};

struct MockIdMap : faiss::IndexIDMap {
    mutable idx_t nCalled{};
    mutable const float *xCalled{};
    mutable int kCalled{};
    mutable float radiusCalled{};
    mutable float *distancesCalled{};
    mutable idx_t *labelsCalled{};
    mutable const faiss::SearchParametersHNSW *paramsCalled{};
    mutable faiss::RangeSearchResult *resCalled{};

    explicit MockIdMap(MockIndex *index) : faiss::IndexIDMapTemplate<faiss::Index>(index) {
    }

    void search(
        idx_t n,
        const float *x,
        idx_t k,
        float *distances,
        idx_t *labels,
        const faiss::SearchParameters *params) const override {
        nCalled = n;
        xCalled = x;
        kCalled = k;
        distancesCalled = distances;
        labelsCalled = labels;
        paramsCalled = dynamic_cast<const faiss::SearchParametersHNSW *>(params);
    }

    void range_search(
        idx_t n,
        const float *x,
        float radius,
        faiss::RangeSearchResult *res,
        const faiss::SearchParameters *params) const override {
        nCalled = n;
        xCalled = x;
        radiusCalled = radius;
        resCalled = res;
        paramsCalled = dynamic_cast<const faiss::SearchParametersHNSW *>(params);
    }

    void resetMock() const {
        nCalled = 0;
        xCalled = nullptr;
        kCalled = 0;
        radiusCalled = 0.0;
        distancesCalled = nullptr;
        labelsCalled = nullptr;
        resCalled = nullptr;
        paramsCalled = nullptr;
    }
};

struct QueryIndexHNSWTestInput {
    std::string description;
    int k;
    int efSearch;
    int filterIdType;
    bool filterIdsPresent;
    bool parentIdsPresent;
};

struct RangeSearchTestInput {
    std::string description;
    float radius;
    int efSearch;
    int filterIdType;
    bool filterIdsPresent;
    bool parentIdsPresent;
};

class FaissWrappeterParametrizedTestFixture : public testing::TestWithParam<QueryIndexHNSWTestInput> {
public:
    FaissWrappeterParametrizedTestFixture() : index_(3), id_map_(&index_) {
        index_.hnsw.efSearch = 100; // assigning 100 to make sure default of 16 is not used anywhere
    }

protected:
    MockIndex index_;
    MockIdMap id_map_;
};

class FaissWrapperParametrizedRangeSearchTestFixture : public testing::TestWithParam<RangeSearchTestInput> {
public:
    FaissWrapperParametrizedRangeSearchTestFixture() : index_(3), id_map_(&index_) {
        index_.hnsw.efSearch = 100; // assigning 100 to make sure default of 16 is not used anywhere
    }

protected:
    MockIndex index_;
    MockIdMap id_map_;
};

namespace query_index_test {

    std::unordered_map<std::string, jobject> methodParams;

    TEST_P(FaissWrappeterParametrizedTestFixture, QueryIndexHNSWTests) {
        // Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        QueryIndexHNSWTestInput const &input = GetParam();
        std::cout << "Running test: " << input.description << std::endl;
        float query[] = {1.2, 2.3, 3.4};

        int efSearch = input.efSearch;
        int expectedEfSearch = 100; //default set in mock
        std::unordered_map<std::string, jobject> methodParams;
        if (efSearch != -1) {
            expectedEfSearch = input.efSearch;
            methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
        }

        std::vector<int> *parentIdPtr = nullptr;
        if (input.parentIdsPresent) {
            std::vector<int> parentId;
            parentId.reserve(2);
            parentId.push_back(1);
            parentId.push_back(2);
            parentIdPtr = &parentId;

            EXPECT_CALL(mockJNIUtil,
                        GetJavaIntArrayLength(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr)))
                    .WillOnce(testing::Return(parentId.size()));

            EXPECT_CALL(mockJNIUtil,
                        GetIntArrayElements(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr), nullptr))
                    .WillOnce(testing::Return(new int[2]{1, 2}));
        }

        // When
        knn_jni::faiss_wrapper::QueryIndex(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&id_map_),
            reinterpret_cast<jfloatArray>(&query), input.k, reinterpret_cast<jobject>(&methodParams),
            reinterpret_cast<jintArray>(parentIdPtr));

        // Then
        int actualEfSearch = id_map_.paramsCalled->efSearch;
        // Asserting the captured argument
        EXPECT_EQ(input.k, id_map_.kCalled);
        EXPECT_EQ(expectedEfSearch, actualEfSearch);
        if (input.parentIdsPresent) {
            faiss::IDGrouper *grouper = id_map_.paramsCalled->grp;
            EXPECT_TRUE(grouper != nullptr);
        }

        id_map_.resetMock();
    }

    INSTANTIATE_TEST_CASE_P(
        QueryIndexHNSWTests,
        FaissWrappeterParametrizedTestFixture,
        ::testing::Values(
            QueryIndexHNSWTestInput{"algoParams present, parent absent", 10, 200, 0, false, false},
            QueryIndexHNSWTestInput{"algoParams absent, parent absent", 10, -1, 0, false, false},
            QueryIndexHNSWTestInput{"algoParams present, parent present", 10, 200, 0, false, true},
            QueryIndexHNSWTestInput{"algoParams absent, parent present", 10, -1, 0, false, true}
        )
    );
}

namespace query_index_with_filter_test {

    TEST_P(FaissWrappeterParametrizedTestFixture, QueryIndexWithFilterHNSWTests) {
        // Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        QueryIndexHNSWTestInput const &input = GetParam();
        float query[] = {1.2, 2.3, 3.4};

        std::vector<int> *parentIdPtr = nullptr;
        if (input.parentIdsPresent) {
            std::vector<int> parentId;
            parentId.reserve(2);
            parentId.push_back(1);
            parentId.push_back(2);
            parentIdPtr = &parentId;

            EXPECT_CALL(mockJNIUtil,
                        GetJavaIntArrayLength(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr)))
                    .WillOnce(testing::Return(parentId.size()));

            EXPECT_CALL(mockJNIUtil,
                        GetIntArrayElements(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr), nullptr))
                    .WillOnce(testing::Return(new int[2]{1, 2}));
        }

        std::vector<long> *filterptr = nullptr;
        if (input.filterIdsPresent) {
            std::vector<long> filter;
            filter.reserve(2);
            filter.push_back(1);
            filter.push_back(2);
            filterptr = &filter;
        }

        int efSearch = input.efSearch;
        int expectedEfSearch = 100; //default set in mock
        std::unordered_map<std::string, jobject> methodParams;
        if (efSearch != -1) {
            expectedEfSearch = input.efSearch;
            methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
        }

        // When
        knn_jni::faiss_wrapper::QueryIndex_WithFilter(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&id_map_),
            reinterpret_cast<jfloatArray>(&query), input.k, reinterpret_cast<jobject>(&methodParams),
            reinterpret_cast<jlongArray>(filterptr),
            input.filterIdType,
            reinterpret_cast<jintArray>(parentIdPtr));

        // Then
        int actualEfSearch = id_map_.paramsCalled->efSearch;
        // Asserting the captured argument
        EXPECT_EQ(input.k, id_map_.kCalled);
        EXPECT_EQ(expectedEfSearch, actualEfSearch);
        if (input.parentIdsPresent) {
            faiss::IDGrouper *grouper = id_map_.paramsCalled->grp;
            EXPECT_TRUE(grouper != nullptr);
        }
        if (input.filterIdsPresent) {
            faiss::IDSelector *sel = id_map_.paramsCalled->sel;
            EXPECT_TRUE(sel != nullptr);
        }
        id_map_.resetMock();
    }

    INSTANTIATE_TEST_CASE_P(
        QueryIndexWithFilterHNSWTests,
        FaissWrappeterParametrizedTestFixture,
        ::testing::Values(
            QueryIndexHNSWTestInput{"algoParams present, parent absent, filter absent", 10, 200, 0, false, false},
            QueryIndexHNSWTestInput{"algoParams present, parent absent, filter absent, filter type 1", 10,  200, 1, false, false},
            QueryIndexHNSWTestInput{"algoParams absent, parent absent, filter present", 10, -1, 0, true, false},
            QueryIndexHNSWTestInput{"algoParams absent, parent absent, filter present, filter type 1", 10, -1, 1, true, false},
            QueryIndexHNSWTestInput{"algoParams present, parent present, filter absent", 10, 200, 0, false, true},
            QueryIndexHNSWTestInput{"algoParams present, parent present, filter absent, filter type 1", 10, 150, 1, false, true},
            QueryIndexHNSWTestInput{"algoParams absent, parent present, filter present", 10, -1, 0, true, true},
            QueryIndexHNSWTestInput{"algoParams absent, parent present, filter present, filter type 1",10, -1, 1, true, true}
        )
    );
}

namespace range_search_test {

    TEST_P(FaissWrapperParametrizedRangeSearchTestFixture, RangeSearchHNSWTests) {
        // Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        RangeSearchTestInput const &input = GetParam();
        float query[] = {1.2, 2.3, 3.4};
        float radius = input.radius;
        int maxResultWindow = 100; // Set your max result window

        std::unordered_map<std::string, jobject> methodParams;
        int efSearch = input.efSearch;
        int expectedEfSearch = 100; // default set in mock
        if (efSearch != -1) {
            expectedEfSearch = input.efSearch;
            methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
        }

        std::vector<int> *parentIdPtr = nullptr;
        if (input.parentIdsPresent) {
            std::vector<int> parentId;
            parentId.reserve(2);
            parentId.push_back(1);
            parentId.push_back(2);
            parentIdPtr = &parentId;

            EXPECT_CALL(mockJNIUtil,
                        GetJavaIntArrayLength(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr)))
                    .WillOnce(testing::Return(parentId.size()));

            EXPECT_CALL(mockJNIUtil,
                        GetIntArrayElements(
                            jniEnv, reinterpret_cast<jintArray>(parentIdPtr), nullptr))
                    .WillOnce(testing::Return(new int[2]{1, 2}));
        }

        std::vector<long> filter;
        std::vector<long> *filterptr = nullptr;
        if (input.filterIdsPresent) {
            filter.reserve(2);
            filter.push_back(1);
            filter.push_back(2);
            filterptr = &filter;
        }

        // When
        knn_jni::faiss_wrapper::RangeSearchWithFilter(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&id_map_),
            reinterpret_cast<jfloatArray>(&query), radius, reinterpret_cast<jobject>(&methodParams),
            maxResultWindow,
            reinterpret_cast<jlongArray>(filterptr),
            input.filterIdType,
            reinterpret_cast<jintArray>(parentIdPtr));

        // Then
        int actualEfSearch = id_map_.paramsCalled->efSearch;
        // Asserting the captured argument
        EXPECT_EQ(expectedEfSearch, actualEfSearch);
        if (input.parentIdsPresent) {
            faiss::IDGrouper *grouper = id_map_.paramsCalled->grp;
            EXPECT_TRUE(grouper != nullptr);
        }
        if (input.filterIdsPresent) {
            faiss::IDSelector *sel = id_map_.paramsCalled->sel;
            EXPECT_TRUE(sel != nullptr);
        }
        id_map_.resetMock();
    }

    INSTANTIATE_TEST_CASE_P(
        RangeSearchHNSWTests,
        FaissWrapperParametrizedRangeSearchTestFixture,
        ::testing::Values(
            RangeSearchTestInput{"algoParams present, parent absent, filter absent", 10.0f, 200, 0, false, false},
            RangeSearchTestInput{"algoParams present, parent absent, filter absent, filter type 1", 10.0f, 200, 1, false, false},
            RangeSearchTestInput{"algoParams absent, parent absent, filter present", 10.0f, -1, 0, true, false},
            RangeSearchTestInput{"algoParams absent, parent absent, filter present, filter type 1", 10.0f, -1, 1, true, false},
            RangeSearchTestInput{"algoParams present, parent present, filter absent", 10.0f, 200, 0, false, true},
            RangeSearchTestInput{"algoParams present, parent present, filter absent, filter type 1", 10.0f, 150, 1, false, true},
            RangeSearchTestInput{"algoParams absent, parent present, filter present", 10.0f, -1, 0, true, true},
            RangeSearchTestInput{"algoParams absent, parent present, filter present, filter type 1", 10.0f, -1, 1, true, true}
        )
    );
}

