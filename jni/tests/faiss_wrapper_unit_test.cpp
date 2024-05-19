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

using ::testing::NiceMock;

using idx_t = faiss::idx_t;

struct MockIndex : faiss::IndexHNSW {
    explicit MockIndex(idx_t d) : faiss::IndexHNSW(d, 32) {
    }
};


struct MockIdMap : faiss::IndexIDMap {
    mutable idx_t nCalled;
    mutable const float *xCalled;
    mutable idx_t kCalled;
    mutable float *distancesCalled;
    mutable idx_t *labelsCalled;
    mutable const faiss::SearchParametersHNSW *paramsCalled;

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

    void resetMock() const {
        nCalled = 0;
        xCalled = nullptr;
        kCalled = 0;
        distancesCalled = nullptr;
        labelsCalled = nullptr;
        paramsCalled = nullptr;
    }
};

struct QueryIndexTestInput {
    string description;
    int k;
    int efSearch;
    int filterIdType;
    bool filterIdsPresent;
    bool parentIdsPresent;
};

class FaissWrappeterParametrizedTestFixture : public testing::TestWithParam<QueryIndexTestInput> {
public:
    FaissWrappeterParametrizedTestFixture() : index_(10), id_map_(&index_) {
        index_.hnsw.efSearch = 100; // assigning 100 to make sure default of 16 is not used anywhere
    };

protected:
    MockIndex index_;
    MockIdMap id_map_;
};

namespace query_index_test {

    TEST_P(FaissWrappeterParametrizedTestFixture, QueryIndexTests) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;


        QueryIndexTestInput const &input = GetParam();
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

        int expectedEfSearch = input.efSearch == -1 ? 100 : input.efSearch; //16 is the default created in mock index

        // When
        knn_jni::faiss_wrapper::QueryIndex(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&id_map_),
            reinterpret_cast<jfloatArray>(&query), input.k, input.efSearch,
            reinterpret_cast<jintArray>(parentIdPtr));

        //Then
        int actualEfSearch = id_map_.paramsCalled->efSearch;
        // Asserting the captured argument
        ASSERT_EQ(input.k, id_map_.kCalled);
        EXPECT_EQ(expectedEfSearch, actualEfSearch);
        if (input.parentIdsPresent) {
            faiss::IDGrouper *grouper = id_map_.paramsCalled->grp;
            EXPECT_TRUE(grouper != nullptr);
        }

        id_map_.resetMock();
    }

    INSTANTIATE_TEST_CASE_P(
        QueryIndexTests,
        FaissWrappeterParametrizedTestFixture,
        ::testing::Values(
            QueryIndexTestInput{"efsearch present, parent absent", 10, 200, 0, false, false},
            QueryIndexTestInput{"efsearch absent, parent absent", 10, -1, 0, false, false},
            QueryIndexTestInput{"efsearch present, parent present", 10, 128, 0, false, true},
            QueryIndexTestInput{"efsearch absent, parent present", 10, -1, 0, false, true}
        )
    );
}

namespace query_index_with_filter_test {

    TEST_P(FaissWrappeterParametrizedTestFixture, QueryIndexWithFilterTests) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;


        QueryIndexTestInput const &input = GetParam();
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

        int expectedEfSearch = input.efSearch == -1 ? 100 : input.efSearch; //16 is the default created in mock index

        // When
        knn_jni::faiss_wrapper::QueryIndex_WithFilter(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&id_map_),
            reinterpret_cast<jfloatArray>(&query), input.k, input.efSearch,
            reinterpret_cast<jlongArray>(filterptr),
            input.filterIdType,
            reinterpret_cast<jintArray>(parentIdPtr));

        //Then
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
        QueryIndexWithFilterTests,
        FaissWrappeterParametrizedTestFixture,
        ::testing::Values(
            QueryIndexTestInput{"efsearch present, parent absent, filter absent", 10, 200, 0, false, false},
            QueryIndexTestInput{"efsearch present, parent absent, filter absent, filter type 1", 10, 200, 1, false, false},
            QueryIndexTestInput{"efsearch absent, parent absent, filter present", 10, -1, 0, true, false},
            QueryIndexTestInput{"efsearch absent, parent absent, filter present, filter type 1", 10, -1, 1, true, false},
            QueryIndexTestInput{"efsearch present, parent present, filter absent", 10, 128, 0, false, true},
            QueryIndexTestInput{"efsearch present, parent present, filter absent, filter type 1", 10, 128, 1, false, true},
            QueryIndexTestInput{"efsearch absent, parent present, filter present", 10, -1, 0, true, true},
            QueryIndexTestInput{"efsearch absent, parent present, filter present, filter type 1",10, -1, 1, true, true}
        )
    );
}
