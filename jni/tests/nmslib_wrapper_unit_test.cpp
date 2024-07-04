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
#include "hnswquery.h"
#include "knnquery.h"
#include "nmslib_wrapper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "jni.h"
#include "test_util.h"
#include "method/hnsw.h"
#include "space/space_dummy.h"

namespace nmslib_query_index_test {

    using ::testing::NiceMock;

    struct QueryIndexHNSWTestInput {
        string description;
        int k;
        int efSearch;
        bool expectedHNSWQuery;
    };

    struct MockNMSIndex : similarity::Hnsw<float> {
        mutable int kCalled;
        mutable int efCalled = -1;

        explicit MockNMSIndex(const similarity::Space<float> &space, const similarity::ObjectVector &data): Hnsw(false,
                space, data) {
            std::vector<string> input;
            input.emplace_back("ef=10");
            similarity::AnyParams ef(input);
            this->Hnsw::SetQueryTimeParams(ef);
        }

        void Search(similarity::KNNQuery<float> *query, similarity::IdType id) const override {
            auto hnsw = dynamic_cast<similarity::HNSWQuery<float> *>(query);
            if (hnsw != nullptr) {
                kCalled = hnsw->GetK();
                efCalled = hnsw->getEf();
            } else {
                kCalled = query->GetK();
            }
            similarity::Object object(5, 0, 3*sizeof(float), new float[] { 2.2f, 2.5f, 2.6f });
            similarity::Object* objectPtr = &object;
            bool added = query->CheckAndAddToResult(0.0f, objectPtr);
        };

        void resetMocks() const {
            kCalled = -1;
            efCalled = -1;
        }
    };

    class NmslibWrapperParametrizedTestFixture : public testing::TestWithParam<QueryIndexHNSWTestInput> {
    public:
        NmslibWrapperParametrizedTestFixture() : space_(nullptr), index_(nullptr) {
            similarity::initLibrary();
            std::string spaceType = knn_jni::L2;
            space_ = similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    spaceType, similarity::AnyParams());
            index_ = new MockNMSIndex(*space_, similarity::ObjectVector());
        };

    protected:
        MockNMSIndex* index_;
        similarity::Space<float>* space_; // Moved from local to member variable
    };


    TEST_P(NmslibWrapperParametrizedTestFixture, QueryIndexHNSWTests) {
        //Given
        JNIEnv *jniEnv = nullptr;
        NiceMock<test_util::MockJNIUtil> mockJNIUtil;


        QueryIndexHNSWTestInput const &input = GetParam();
        float query[] = { 1.2f, 2.3f, 3.4f };

        std::string spaceType = knn_jni::L2;
        std::unique_ptr<knn_jni::nmslib_wrapper::IndexWrapper> indexWrapper(
            new knn_jni::nmslib_wrapper::IndexWrapper(spaceType));
        indexWrapper->index.reset(index_);

        int efSearch = input.efSearch;
        std::unordered_map<std::string, jobject> methodParams;
        if (efSearch != -1) {
            methodParams[knn_jni::EF_SEARCH] = reinterpret_cast<jobject>(&efSearch);
        }
        EXPECT_CALL(mockJNIUtil,
                        GetJavaFloatArrayLength(
                            jniEnv, reinterpret_cast<jfloatArray>(query)))
                    .WillOnce(testing::Return(3));

        EXPECT_CALL(mockJNIUtil,
                    ReleaseFloatArrayElements(
                        jniEnv, reinterpret_cast<jfloatArray>(query), query, JNI_ABORT));
        EXPECT_CALL(mockJNIUtil,
                    GetFloatArrayElements(
                        jniEnv, reinterpret_cast<jfloatArray>(query), nullptr))
                .WillOnce(testing::Return(query));

        knn_jni::nmslib_wrapper::QueryIndex(
            &mockJNIUtil, jniEnv,
            reinterpret_cast<jlong>(indexWrapper.get()),
            reinterpret_cast<jfloatArray>(&query), input.k, reinterpret_cast<jobject>(&methodParams));

        if (input.expectedHNSWQuery) {
            EXPECT_EQ(input.efSearch, index_->efCalled);
            EXPECT_EQ(input.k, index_->kCalled);
        } else {
            EXPECT_EQ(input.k, index_->kCalled);
        }
        index_->resetMocks();
    }

    INSTANTIATE_TEST_CASE_P(
        QueryIndexHNSWTests,
        NmslibWrapperParametrizedTestFixture,
        ::testing::Values(
            QueryIndexHNSWTestInput{"methodParams present", 10, 200, true},
            QueryIndexHNSWTestInput{"methodParams absent", 5, -1, false }
        )
    );
}
