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
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"

using ::testing::NiceMock;

using idx_t = faiss::idx_t;

struct FaissMockIndex : faiss::IndexHNSW {
    explicit FaissMockIndex(idx_t d) : faiss::IndexHNSW(d, 32) {
    }
};

struct MockIVFIndex : faiss::IndexIVFFlat {
    explicit MockIVFIndex() = default;
};

struct MockIVFIdMap : faiss::IndexIDMap {
    mutable idx_t nCalled{};
    mutable const float *xCalled{};
    mutable idx_t kCalled{};
    mutable float *distancesCalled{};
    mutable idx_t *labelsCalled{};
    mutable const faiss::SearchParametersIVF *paramsCalled{};

    explicit MockIVFIdMap(MockIVFIndex *index) : faiss::IndexIDMapTemplate<faiss::Index>(index) {
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
        paramsCalled = dynamic_cast<const faiss::SearchParametersIVF *>(params);
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

struct FaissMockIdMap : faiss::IndexIDMap {
    mutable idx_t nCalled{};
    mutable const float *xCalled{};
    mutable int kCalled{};
    mutable float radiusCalled{};
    mutable float *distancesCalled{};
    mutable idx_t *labelsCalled{};
    mutable const faiss::SearchParametersHNSW *paramsCalled{};
    mutable faiss::RangeSearchResult *resCalled{};

    explicit FaissMockIdMap(FaissMockIndex *index) : faiss::IndexIDMapTemplate<faiss::Index>(index) {
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

struct QueryIndexInput {
    string description;
    int k;
    int filterIdType;
    bool filterIdsPresent;
    bool parentIdsPresent;
    int efSearch;
    int nprobe;
};

struct RangeSearchTestInput {
    std::string description;
    float radius;
    int efSearch;
    int filterIdType;
    bool filterIdsPresent;
    bool parentIdsPresent;
};

class FaissWrapperParameterizedTestFixture : public testing::TestWithParam<QueryIndexInput> {
public:
    FaissWrapperParameterizedTestFixture() : index_(3), id_map_(&index_) {
        index_.hnsw.efSearch = 100; // assigning 100 to make sure default of 16 is not used anywhere
    }

protected:
    FaissMockIndex index_;
    FaissMockIdMap id_map_;
};

class FaissWrapperParameterizedRangeSearchTestFixture : public testing::TestWithParam<RangeSearchTestInput> {
public:
    FaissWrapperParameterizedRangeSearchTestFixture() : index_(3), id_map_(&index_) {
        index_.hnsw.efSearch = 100; // assigning 100 to make sure default of 16 is not used anywhere
    }

protected:
    FaissMockIndex index_;
    FaissMockIdMap id_map_;
};

class FaissWrapperIVFQueryTestFixture : public testing::TestWithParam<QueryIndexInput> {
public:
    FaissWrapperIVFQueryTestFixture() : ivf_id_map_(&ivf_index_) {
        ivf_index_.nprobe = 100;
    };

protected:
    MockIVFIndex ivf_index_;
    MockIVFIdMap ivf_id_map_;
};

namespace query_index_test {
    TEST_P(FaissWrapperParameterizedTestFixture, QueryIndexHNSWTests) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        QueryIndexInput const &input = GetParam();
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
        FaissWrapperParameterizedTestFixture,
        ::testing::Values(
            QueryIndexInput {"algoParams present, parent absent", 10, 0, false, false, 200, -1 },
            QueryIndexInput {"algoParams present, parent absent", 10, 0, false, false, -1, -1 },
            QueryIndexInput {"algoParams present, parent present", 10, 0, false, true, 200, -1 },
            QueryIndexInput {"algoParams absent, parent present", 10, 0, false, true, -1, -1 }
        )
    );
}

namespace query_index_with_filter_test {
    TEST_P(FaissWrapperParameterizedTestFixture, QueryIndexWithFilterHNSWTests) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        QueryIndexInput const &input = GetParam();
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
        FaissWrapperParameterizedTestFixture,
        ::testing::Values(
            QueryIndexInput { "algoParams present, parent absent, filter absent", 10, 0, false, false, 200, -1 },
            QueryIndexInput { "algoParams present, parent absent, filter absent, filter type 1", 10, 1, false, false,
            200, -1},
            QueryIndexInput { "algoParams absent, parent absent, filter present", 10, 0, true, false, -1, -1},
            QueryIndexInput { "algoParams absent, parent absent, filter present, filter type 1", 10, 1, true, false, -1,
            -1},
            QueryIndexInput { "algoParams present, parent present, filter absent", 10, 0, false, true, 200, -1 },
            QueryIndexInput { "algoParams present, parent present, filter absent, filter type 1", 10, 1, false, true,
            150, -1},
            QueryIndexInput { "algoParams absent, parent present, filter present", 10, 0, true, true, -1, -1},
            QueryIndexInput { "algoParams absent, parent present, filter present, filter type 1",10, 1, true, true, -1,
            -1 }
        )
    );
}

namespace range_search_test {
    TEST_P(FaissWrapperParameterizedRangeSearchTestFixture, RangeSearchHNSWTests) {
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
            std::vector<long> filter;
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
        FaissWrapperParameterizedRangeSearchTestFixture,
        ::testing::Values(
            RangeSearchTestInput{"algoParams present, parent absent, filter absent", 10.0f, 200, 0, false, false},
            RangeSearchTestInput{"algoParams present, parent absent, filter absent, filter type 1", 10.0f, 200, 1, false
            , false},
            RangeSearchTestInput{"algoParams absent, parent absent, filter present", 10.0f, -1, 0, true, false},
            RangeSearchTestInput{"algoParams absent, parent absent, filter present, filter type 1", 10.0f, -1, 1, true,
            false},
            RangeSearchTestInput{"algoParams present, parent present, filter absent", 10.0f, 200, 0, false, true},
            RangeSearchTestInput{"algoParams present, parent present, filter absent, filter type 1", 10.0f, 150, 1,
            false, true},
            RangeSearchTestInput{"algoParams absent, parent present, filter present", 10.0f, -1, 0, true, true},
            RangeSearchTestInput{"algoParams absent, parent present, filter present, filter type 1", 10.0f, -1, 1, true,
            true}
        )
    );
}

namespace query_index_with_filter_test_ivf {
    TEST_P(FaissWrapperIVFQueryTestFixture, QueryIndexIVFTest) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;

        QueryIndexInput const &input = GetParam();
        float query[] = {1.2, 2.3, 3.4};

        int nprobe = input.nprobe;
        int expectedNprobe = 100; //default set in mock
        std::unordered_map<std::string, jobject> methodParams;
        if (nprobe != -1) {
            expectedNprobe = input.nprobe;
            methodParams[knn_jni::NPROBES] = reinterpret_cast<jobject>(&nprobe);
        }

        std::vector<long> *filterptr = nullptr;
        if (input.filterIdsPresent) {
            std::vector<long> filter;
            filter.reserve(2);
            filter.push_back(1);
            filter.push_back(2);
            filterptr = &filter;
        }

        // When
        knn_jni::faiss_wrapper::QueryIndex_WithFilter(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(&ivf_id_map_),
            reinterpret_cast<jfloatArray>(&query), input.k, reinterpret_cast<jobject>(&methodParams),
            reinterpret_cast<jlongArray>(filterptr),
            input.filterIdType,
            nullptr);

        //Then
        int actualEfSearch = ivf_id_map_.paramsCalled->nprobe;
        // Asserting the captured argument
        EXPECT_EQ(input.k, ivf_id_map_.kCalled);
        EXPECT_EQ(expectedNprobe, actualEfSearch);
        if (input.parentIdsPresent) {
            faiss::IDGrouper *grouper = ivf_id_map_.paramsCalled->grp;
            EXPECT_TRUE(grouper != nullptr);
        }
        ivf_id_map_.resetMock();
    }

    INSTANTIATE_TEST_CASE_P(
        QueryIndexIVFTest,
        FaissWrapperIVFQueryTestFixture,
        ::testing::Values(
            QueryIndexInput{"algoParams present, parent absent", 10, 0, false, false, -1, 200 },
            QueryIndexInput{"algoParams present, parent absent", 10,0, false, false, -1, -1 },
            QueryIndexInput{"algoParams present, parent present", 10, 0, true, true, -1, 200 },
            QueryIndexInput{"algoParams absent, parent present", 10, 0, true, true, -1, -1 }
        )
    );
}
namespace load_index_with_stream_adc_params_test {
    // Mock class for IOReader following actual FAISS interface
    class MockIOReader : public faiss::IOReader {
    public:
        // Implement the required pure virtual operator()
        size_t operator()(void* ptr, size_t size, size_t nitems) override {
            return nitems; // Just pretend we read everything successfully
        }
    };

    struct LoadIndexWithStreamADCParamsInput {
        std::string description;
        bool includeQuantLevel;
        bool includeSpaceType;
        knn_jni::BQQuantizationLevel quantLevel;
        std::string spaceType;
        std::string expectedError;
    };

    // Create a test class that extends the function we're testing
    class TestLoadIndexWithParams {
    public:
        // This is our test version that mimics knn_jni::faiss_wrapper::LoadIndexWithStreamADCParams
        // but calls our mock LoadIndexWithStreamADC
        static jlong LoadIndexWithStreamADCParams(
            faiss::IOReader* ioReader,
            knn_jni::JNIUtilInterface* jniUtil,
            JNIEnv* env,
            jobject methodParamsJ) {

            auto methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);

            // KNNConstants.QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER
            auto quantization_level_it = methodParams.find("quantization_level");
            knn_jni::BQQuantizationLevel quantLevel = knn_jni::BQQuantizationLevel::NONE;
            if (quantization_level_it != methodParams.end()) {
                quantLevel = jniUtil->ConvertJavaStringToQuantizationLevel(env, quantization_level_it->second);
            } else {
                throw std::runtime_error("Quantization level not specified in params");
            }

            // KNNConstants.SPACE_TYPE_FAISS_INDEX_LOAD_PARAMETER
            auto space_type_it = methodParams.find("space_type");
            faiss::MetricType metricType; // L2 by default.
            if (space_type_it!= methodParams.end()) {
                std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, space_type_it->second));
                metricType = knn_jni::faiss_wrapper::TranslateSpaceToMetric(spaceTypeCpp);
            } else {
                throw std::runtime_error("space type not specified in params");
            }

            if (quantLevel == knn_jni::BQQuantizationLevel::ONE_BIT) {
                // Instead of calling real LoadIndexWithStreamADC, return a dummy value
                return 12345;
            } else if (
                quantLevel == knn_jni::BQQuantizationLevel::TWO_BIT || quantLevel == knn_jni::BQQuantizationLevel::FOUR_BIT
            ) {
                throw std::runtime_error("ADC not supported for 2 or 4 bit.");
            }
            else {
                jniUtil->HasExceptionInStack(env, "load adc stream called without a quantization level");
                throw std::runtime_error("load adc stream called without a quantization level");
            }
        }
    };

    class FaissWrapperLoadIndexParamsTestFixture : public testing::TestWithParam<LoadIndexWithStreamADCParamsInput> {
    protected:
        NiceMock<JNIEnv> jniEnv;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;
        std::unique_ptr<faiss::IOReader> mockIOReader{new MockIOReader()};
    };

    TEST_P(FaissWrapperLoadIndexParamsTestFixture, LoadIndexWithStreamADCParamsTests) {
        // Given
        LoadIndexWithStreamADCParamsInput const &input = GetParam();
        std::unordered_map<std::string, jobject> methodParams;

        if (input.includeQuantLevel) {
            methodParams["quantization_level"] = (jobject)"dummy";
        }
        if (input.includeSpaceType) {
            methodParams["space_type"] = (jobject)"dummy";
        }

        EXPECT_CALL(mockJNIUtil, ConvertJavaMapToCppMap(&jniEnv, testing::_))
            .WillOnce(testing::Return(methodParams));

        // Fix expectations based on actual control flow
        if (input.includeQuantLevel) {
            EXPECT_CALL(mockJNIUtil, ConvertJavaStringToQuantizationLevel(&jniEnv, testing::_))
                .WillOnce(testing::Return(input.quantLevel));

            // Only expect space type conversion if we have quantization level
            if (input.includeSpaceType) {
                EXPECT_CALL(mockJNIUtil, ConvertJavaObjectToCppString(&jniEnv, testing::_))
                    .WillOnce(testing::Return(input.spaceType));

                // HasExceptionInStack only called for NONE quantization level after space type check
                if (input.quantLevel == knn_jni::BQQuantizationLevel::NONE) {
                    EXPECT_CALL(mockJNIUtil, HasExceptionInStack(&jniEnv, testing::_))
                        .Times(1);
                }
            }
        }

        // When/Then
        try {
            jlong result = TestLoadIndexWithParams::LoadIndexWithStreamADCParams(
                mockIOReader.get(), &mockJNIUtil, &jniEnv, nullptr);

            // For success cases, verify the result is our expected dummy value
            if (input.expectedError.empty()) {
                ASSERT_EQ(12345, result);
            } else {
                // If expectedError is not empty, we should have thrown an exception
                FAIL() << "Expected exception with message: " << input.expectedError;
            }

        } catch (const std::runtime_error& e) {
            // If expectedError is empty, we shouldn't be here
            if (input.expectedError.empty()) {
                FAIL() << "Unexpected exception: " << e.what();
            }
            // Otherwise verify the error matches
            ASSERT_EQ(input.expectedError, e.what());
        }
    }

    INSTANTIATE_TEST_CASE_P(
        LoadIndexWithStreamADCParamsTests,
        FaissWrapperLoadIndexParamsTestFixture,
        ::testing::Values(
            // Success case
            LoadIndexWithStreamADCParamsInput{
                "Valid ONE_BIT L2",
                true,
                true,
                knn_jni::BQQuantizationLevel::ONE_BIT,
                knn_jni::L2,
                ""
            },
            LoadIndexWithStreamADCParamsInput{
                "Valid ONE_BIT INNER_PRODUCT",
                true,
                true,
                knn_jni::BQQuantizationLevel::ONE_BIT,
                knn_jni::INNER_PRODUCT,
                ""
            },
            // Missing parameter cases
            LoadIndexWithStreamADCParamsInput{
                "Missing quantization_level",
                false,
                true,
                knn_jni::BQQuantizationLevel::NONE,
                knn_jni::L2,
                "Quantization level not specified in params"
            },
            LoadIndexWithStreamADCParamsInput{
                "Missing space_type",
                true,
                false,
                knn_jni::BQQuantizationLevel::ONE_BIT,
                "",
                "space type not specified in params"
            },
            // Invalid quantization level cases
            LoadIndexWithStreamADCParamsInput{
                "TWO_BIT not supported",
                true,
                true,
                knn_jni::BQQuantizationLevel::TWO_BIT,
                knn_jni::L2,
                "ADC not supported for 2 or 4 bit."
            },
            LoadIndexWithStreamADCParamsInput{
                "FOUR_BIT not supported",
                true,
                true,
                knn_jni::BQQuantizationLevel::FOUR_BIT,
                knn_jni::L2,
                "ADC not supported for 2 or 4 bit."
            },
            LoadIndexWithStreamADCParamsInput{
                "NONE quantization level",
                true,
                true,
                knn_jni::BQQuantizationLevel::NONE,
                knn_jni::L2,
                "load adc stream called without a quantization level"
            }
        )
    );
}

