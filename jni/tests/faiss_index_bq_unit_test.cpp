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

#include "FaissIndexBQ.h"

#include <vector>
#include <cmath>
#include <random>
#include <bitset>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jni_util.h"
#include "jni.h"
#include "test_util.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexFlatCodes.h"
#include "faiss/IndexIDMap.h"

using ::testing::NiceMock;
using idx_t = faiss::idx_t;

namespace knn_jni {
namespace faiss_wrapper {

// Helper functions for testing
class TestHelpers {
public:
    static constexpr float NEAR_THRESHOLD = 1e-4;
    // Pack float vector into binary representation (1-bit quantization)
    static std::vector<uint8_t> packBits(const std::vector<float>& vector) {
        int dimension = vector.size();
        assert(dimension % 8 == 0);

        std::vector<uint8_t> packed(dimension / 8, 0);

        for (int i = 0; i < dimension; i += 8) {
            uint8_t byte = 0;
            for (int j = 0; j < 8; j++) {
                // Follow the bit packing strategy mentioned in the code comments
                // Bit position (7-j) to match the scanning pattern
                if (vector[i + j] > 0.0f) {
                    byte |= (1 << (7 - j));
                }
            }
            packed[i / 8] = byte;
        }

        return packed;
    }

    // Compute expected L2 distance between query and binary vector
    static float computeExpectedL2Distance(const std::vector<float>& query,
                                         const std::vector<uint8_t>& binaryCode) {
        float distance = 0.0f;
        int dimension = query.size();

        for (int i = 0; i < dimension; i++) {
            int byteIdx = i / 8;
            int bitIdx = 7 - (i % 8); // Match the bit packing pattern

            float binaryValue = ((binaryCode[byteIdx] >> bitIdx) & 1) ? 1.0f : 0.0f;
            float diff = query[i] - binaryValue;
            distance += diff * diff;
        }

        return distance;
    }

    // Compute expected inner product distance between query and binary vector
    static float computeExpectedInnerProductDistance(const std::vector<float>& query,
                                                   const std::vector<uint8_t>& binaryCode) {
        float innerProduct = 0.0f;
        int dimension = query.size();

        for (int i = 0; i < dimension; i++) {
            int byteIdx = i / 8;
            int bitIdx = 7 - (i % 8); // Match the bit packing pattern

            float binaryValue = ((binaryCode[byteIdx] >> bitIdx) & 1) ? 1.0f : 0.0f;
            innerProduct += query[i] * binaryValue;
        }

        return innerProduct;
    }

    // Generate random float vector
    static std::vector<float> generateRandomVector(int dimension, float min = -1.0f, float max = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);

        std::vector<float> vector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector[i] = dis(gen);
        }

        return vector;
    }

    // Generate specific test vectors for deterministic testing
    static std::vector<float> generateTestVector(int dimension, int pattern) {
        std::vector<float> vector(dimension);

        switch (pattern) {
            case 0: // All positive
                std::fill(vector.begin(), vector.end(), 0.5f);
                break;
            case 1: // All negative
                std::fill(vector.begin(), vector.end(), -0.5f);
                break;
            case 2: // Alternating
                for (int i = 0; i < dimension; i++) {
                    vector[i] = (i % 2 == 0) ? 0.5f : -0.5f;
                }
                break;
            case 3: // Increasing
                for (int i = 0; i < dimension; i++) {
                    vector[i] = -1.0f + (2.0f * i / dimension);
                }
                break;
            default:
                std::fill(vector.begin(), vector.end(), 0.0f);
        }

        return vector;
    }
};

class ADCFlatCodesDistanceComputerTestFixture : public testing::Test {
protected:
    void SetUp() override {
        dimension_ = 64;
        metricType_ = faiss::METRIC_L2;
    }

    void setupTest(int dimension, faiss::MetricType metricType, int queryPattern, int codePattern) {
        dimension_ = dimension;
        metricType_ = metricType;

        // Generate test query and code vectors
        queryVector_ = TestHelpers::generateTestVector(dimension_, queryPattern);
        codeVector_ = TestHelpers::generateTestVector(dimension_, codePattern);
        packedCode_ = TestHelpers::packBits(codeVector_);

        // Create the distance computer
        computer_ = std::make_unique<ADCFlatCodesDistanceComputer1Bit>(
            packedCode_.data(), packedCode_.size(), dimension_, metricType_);
    }

    int dimension_;
    faiss::MetricType metricType_;
    std::vector<float> queryVector_;
    std::vector<float> codeVector_;
    std::vector<uint8_t> packedCode_;
    std::unique_ptr<ADCFlatCodesDistanceComputer1Bit> computer_;
};

class FaissIndexBQTest : public testing::Test {
protected:
    void SetUp() override {
        dimension_ = 64; // Must be multiple of 8
        metricType_ = faiss::METRIC_L2;

        // Create some test codes
        codes_ = std::vector<uint8_t>(dimension_ / 8 * 3); // 3 vectors
        std::iota(codes_.begin(), codes_.end(), 1); // Fill with 1, 2, 3...

        index_ = std::make_unique<FaissIndexBQ>(dimension_, codes_, metricType_);
    }

    int dimension_;
    faiss::MetricType metricType_;
    std::vector<uint8_t> codes_;
    std::unique_ptr<FaissIndexBQ> index_;
};

//// Test ADCFlatCodesDistanceComputer1Bit constructor
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, ConstructorL2AllPositiveTest) {
//    setupTest(64, faiss::METRIC_L2, 0, 0);
//
//    EXPECT_EQ(computer_->dimension, dimension_);
//    EXPECT_EQ(computer_->code_size, dimension_ / 8);
//    EXPECT_EQ(computer_->metric_type, metricType_);
//    EXPECT_EQ(computer_->codes, packedCode_.data());
//    EXPECT_EQ(computer_->query, nullptr);
//    EXPECT_EQ(computer_->correction_amount, 0.0f);
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, ConstructorInnerProductTest) {
//    setupTest(64, faiss::METRIC_INNER_PRODUCT, 1, 1);
//
//    EXPECT_EQ(computer_->dimension, dimension_);
//    EXPECT_EQ(computer_->code_size, dimension_ / 8);
//    EXPECT_EQ(computer_->metric_type, metricType_);
//    EXPECT_EQ(computer_->codes, packedCode_.data());
//    EXPECT_EQ(computer_->query, nullptr);
//    EXPECT_EQ(computer_->correction_amount, 0.0f);
//}
//
//// Test set_query functionality
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, SetQueryL2Test) {
//    setupTest(64, faiss::METRIC_L2, 0, 0);
//    computer_->set_query(queryVector_.data());
//
//    EXPECT_EQ(computer_->query, queryVector_.data());
//    EXPECT_EQ(computer_->coord_scores.size(), dimension_);
//    EXPECT_EQ(computer_->lookup_table.size(), dimension_ / 8);
//
//    for (const auto& batch : computer_->lookup_table) {
//        EXPECT_EQ(batch.size(), 256); // 2^8 possibilities per batch
//    }
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, SetQueryInnerProductTest) {
//    setupTest(64, faiss::METRIC_INNER_PRODUCT, 2, 2);
//    computer_->set_query(queryVector_.data());
//
//    EXPECT_EQ(computer_->query, queryVector_.data());
//    EXPECT_EQ(computer_->coord_scores.size(), dimension_);
//    EXPECT_EQ(computer_->lookup_table.size(), dimension_ / 8);
//
//    for (const auto& batch : computer_->lookup_table) {
//        EXPECT_EQ(batch.size(), 256); // 2^8 possibilities per batch
//    }
//}
//
//// Test distance computation accuracy
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2AllPositiveTest) {
//    setupTest(64, faiss::METRIC_L2, 0, 0);
//    computer_->set_query(queryVector_.data());
//
//    float computedDistance = computer_->distance_to_code(packedCode_.data());
//    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector_, packedCode_);
//
//    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2AlternatingTest) {
//    setupTest(64, faiss::METRIC_L2, 2, 2);
//    computer_->set_query(queryVector_.data());
//
//    float computedDistance = computer_->distance_to_code(packedCode_.data());
//    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector_, packedCode_);
//
//    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationInnerProductTest) {
//    setupTest(64, faiss::METRIC_INNER_PRODUCT, 0, 1);
//    computer_->set_query(queryVector_.data());
//
//    float computedDistance = computer_->distance_to_code(packedCode_.data());
//    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector_, packedCode_);
//
//    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationSmallDimensionTest) {
//    setupTest(8, faiss::METRIC_L2, 0, 1);
//    computer_->set_query(queryVector_.data());
//
//    float computedDistance = computer_->distance_to_code(packedCode_.data());
//    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector_, packedCode_);
//
//    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
//}
//
//TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationLargeDimensionTest) {
//    setupTest(128, faiss::METRIC_L2, 3, 2);
//    computer_->set_query(queryVector_.data());
//
//    float computedDistance = computer_->distance_to_code(packedCode_.data());
//    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector_, packedCode_);
//
//    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
//}
//
//// Test coordinate scores computation for L2
//TEST(ADCFlatCodesDistanceComputerTest, CoordScoresL2Test) {
//    int dimension = 16;
//    std::vector<uint8_t> dummyCode(dimension / 8, 0);
//    std::vector<float> query = {0.5f, -0.3f, 1.2f, -0.8f, 0.0f, 0.7f, -1.1f, 0.4f,
//                               0.9f, -0.2f, 0.6f, -0.5f, 1.0f, -0.9f, 0.3f, -0.6f};
//
//    ADCFlatCodesDistanceComputer1Bit computer(dummyCode.data(), dummyCode.size(),
//                                             dimension, faiss::METRIC_L2);
//    computer.set_query(query.data());
//
//    // Check coordinate scores: should be 1 - 2*x
//    for (int i = 0; i < dimension; i++) {
//        float expected = 1.0f - 2.0f * query[i];
//        EXPECT_NEAR(computer.coord_scores[i], expected, 1e-6f);
//    }
//
//    // Check correction amount: should be sum of x^2
//    float expectedCorrection = 0.0f;
//    for (float x : query) {
//        expectedCorrection += x * x;
//    }
//    EXPECT_NEAR(computer.correction_amount, expectedCorrection, 1e-6f);
//}
//
//// Test coordinate scores computation for Inner Product
//TEST(ADCFlatCodesDistanceComputerTest, CoordScoresInnerProductTest) {
//    int dimension = 16;
//    std::vector<uint8_t> dummyCode(dimension / 8, 0);
//    std::vector<float> query = {0.5f, -0.3f, 1.2f, -0.8f, 0.0f, 0.7f, -1.1f, 0.4f,
//                               0.9f, -0.2f, 0.6f, -0.5f, 1.0f, -0.9f, 0.3f, -0.6f};
//
//    ADCFlatCodesDistanceComputer1Bit computer(dummyCode.data(), dummyCode.size(),
//                                             dimension, faiss::METRIC_INNER_PRODUCT);
//    computer.set_query(query.data());
//
//    // Check coordinate scores: should be equal to query values
//    for (int i = 0; i < dimension; i++) {
//        EXPECT_NEAR(computer.coord_scores[i], query[i], 1e-6f);
//    }
//
//    // Correction amount should be 0 for inner product
//    EXPECT_NEAR(computer.correction_amount, 0.0f, 1e-6f);
//}
//
//// Test symmetric_dis throws exception
//TEST(ADCFlatCodesDistanceComputerTest, SymmetricDisThrowsTest) {
//    int dimension = 16;
//    std::vector<uint8_t> dummyCode(dimension / 8, 0);
//
//    ADCFlatCodesDistanceComputer1Bit computer(dummyCode.data(), dummyCode.size(),
//                                             dimension, faiss::METRIC_L2);
//
//    EXPECT_THROW(computer.symmetric_dis(0, 1), std::runtime_error);
//}
//
//// Test unsupported metric throws exception
//TEST(ADCFlatCodesDistanceComputerTest, UnsupportedMetricTest) {
//    int dimension = 16;
//    std::vector<uint8_t> dummyCode(dimension / 8, 0);
//    std::vector<float> query(dimension, 0.5f);
//
//    ADCFlatCodesDistanceComputer1Bit computer(dummyCode.data(), dummyCode.size(),
//                                             dimension, faiss::METRIC_Lp); // Unsupported metric
//
//    EXPECT_THROW(computer.set_query(query.data()), std::runtime_error);
//}
//
//// Test FaissIndexBQ constructor
//TEST_F(FaissIndexBQTest, ConstructorTest) {
//    EXPECT_EQ(index_->d, dimension_);
//    EXPECT_EQ(index_->code_size, dimension_ / 8);
//    EXPECT_EQ(index_->metric_type, metricType_);
//    EXPECT_EQ(index_->codes_ptr, &codes_);
//}
//
//// Test FaissIndexBQ init method
//TEST_F(FaissIndexBQTest, InitTest) {
//    // Create mock parent and grandparent indices
//    faiss::IndexFlat parent(dimension_, metricType_);
//    faiss::IndexIDMap grandParent(&parent);
//
//    // Before init, ntotal should be 0
//    EXPECT_EQ(index_->ntotal, 0);
//    EXPECT_EQ(parent.ntotal, 0);
//    EXPECT_EQ(grandParent.ntotal, 0);
//
//    // After init, ntotal should be calculated based on codes size
//    index_->init(&parent, &grandParent);
//
//    idx_t expectedNTotal = codes_.size() / (dimension_ / 8);
//    EXPECT_EQ(index_->ntotal, expectedNTotal);
//    EXPECT_EQ(parent.ntotal, expectedNTotal);
//    EXPECT_EQ(grandParent.ntotal, expectedNTotal);
//}
//
//// Test FaissIndexBQ get_FlatCodesDistanceComputer
//TEST_F(FaissIndexBQTest, GetDistanceComputerTest) {
//    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
//        index_->get_FlatCodesDistanceComputer());
//
//    EXPECT_NE(computer.get(), nullptr);
//
//    // Try to cast to our specific type
//    ADCFlatCodesDistanceComputer1Bit* adcComputer =
//        dynamic_cast<ADCFlatCodesDistanceComputer1Bit*>(computer.get());
//
//    EXPECT_NE(adcComputer, nullptr);
//    EXPECT_EQ(adcComputer->dimension, dimension_);
//    EXPECT_EQ(adcComputer->metric_type, metricType_);
//}
//
//// Test batch lookup computation
//TEST(ADCFlatCodesDistanceComputerTest, BatchLookupTest) {
//    int dimension = 16;
//    std::vector<uint8_t> dummyCode(dimension / 8, 0);
//    std::vector<float> query(dimension, 0.5f);
//
//    ADCFlatCodesDistanceComputer1Bit computer(dummyCode.data(), dummyCode.size(),
//                                             dimension, faiss::METRIC_L2);
//    computer.set_query(query.data());
//
//    // Check that lookup table is properly constructed
//    EXPECT_EQ(computer.lookup_table.size(), 2); // 16/8 = 2 batches
//
//    for (const auto& batch : computer.lookup_table) {
//        EXPECT_EQ(batch.size(), 256);
//        // First element should always be 0 (no bits set)
//        EXPECT_EQ(batch[0], 0.0f);
//    }
//}
//
//// Integration test with multiple vectors
//TEST(ADCFlatCodesDistanceComputerTest, MultipleVectorsTest) {
//    int dimension = 32;
//    int numVectors = 5;
//
//    // Create multiple test vectors
//    std::vector<std::vector<float>> testVectors;
//    std::vector<uint8_t> allCodes;
//
//    for (int i = 0; i < numVectors; i++) {
//        auto vec = TestHelpers::generateTestVector(dimension, i % 4);
//        testVectors.push_back(vec);
//
//        auto packed = TestHelpers::packBits(vec);
//        allCodes.insert(allCodes.end(), packed.begin(), packed.end());
//    }
//
//    std::vector<float> query = TestHelpers::generateRandomVector(dimension);
//
//    ADCFlatCodesDistanceComputer1Bit computer(allCodes.data(), dimension / 8,
//                                             dimension, faiss::METRIC_L2);
//    computer.set_query(query.data());
//
//    // Test distance computation for each vector
//    for (int i = 0; i < numVectors; i++) {
//        const uint8_t* codePtr = allCodes.data() + i * (dimension / 8);
//        float computedDistance = computer.distance_to_code(codePtr);
//
//        auto packedCode = TestHelpers::packBits(testVectors[i]);
//        float expectedDistance = TestHelpers::computeExpectedL2Distance(query, packedCode);
//
//        EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
//            << "Vector " << i << " distance mismatch";
//    }
//}
//

// Test distance computation accuracy using FaissIndexBQ workflow
TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2AllPositiveTest) {
    int dimension = 64;

    // Create test vectors and pack them
    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 0); // all positive
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 0);  // all positive
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    // Create FaissIndexBQ following the actual usage pattern
    std::vector<uint8_t> codes = packedCode; // Copy to simulate the codes storage
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    // Get distance computer from the index
    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());

    // Set query
    computer->set_query(queryVector.data());

    // Compute distance
    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2AllNegativeTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 1); // all negative
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 1);  // all negative
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2AlternatingTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 2); // alternating
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 2);  // alternating
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationL2MixedTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 0); // positive query
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 1);  // negative code
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationInnerProductAllPositiveTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 0); // all positive
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 0);  // all positive
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_INNER_PRODUCT);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationInnerProductMixedTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 0); // positive query
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 1);  // negative code
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_INNER_PRODUCT);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationInnerProductAlternatingTest) {
    int dimension = 64;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 2); // alternating
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 2);  // alternating
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_INNER_PRODUCT);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationSmallDimensionL2Test) {
    int dimension = 8;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 0); // positive query
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 1);  // negative code
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationSmallDimensionInnerProductTest) {
    int dimension = 8;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 1); // negative query
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 0);  // positive code
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_INNER_PRODUCT);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationLargeDimensionL2Test) {
    int dimension = 128;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 3); // increasing
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 2);  // alternating
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

TEST_F(ADCFlatCodesDistanceComputerTestFixture, DistanceComputationLargeDimensionInnerProductTest) {
    int dimension = 128;

    std::vector<float> queryVector = TestHelpers::generateTestVector(dimension, 3); // increasing
    std::vector<float> codeVector = TestHelpers::generateTestVector(dimension, 2);  // alternating
    std::vector<uint8_t> packedCode = TestHelpers::packBits(codeVector);

    std::vector<uint8_t> codes = packedCode;
    FaissIndexBQ index(dimension, codes, faiss::METRIC_INNER_PRODUCT);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(queryVector.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedInnerProductDistance(queryVector, packedCode);

    EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
        << "Computed: " << computedDistance << ", Expected: " << expectedDistance;
}

// Test with specific bit patterns to verify bit extraction logic
TEST(ADCFlatCodesDistanceComputerTest, DistanceComputationBitPatternTest) {
    int dimension = 8;
    std::vector<uint8_t> codes = {0b10101010}; // Alternating bit pattern
    std::vector<float> query = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    FaissIndexBQ index(dimension, codes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(query.data());

    float computedDistance = computer->distance_to_code(codes.data());
    float expectedDistance = TestHelpers::computeExpectedL2Distance(query, codes);

    EXPECT_NEAR(computedDistance, expectedDistance, 1e-6f);

    // Manual calculation for verification
    // Bit pattern 10101010 means: 1,0,1,0,1,0,1,0 (reading left to right)
    float manualDistance = 0.0f;
    std::vector<int> expectedBits = {1, 0, 1, 0, 1, 0, 1, 0};
    for (int i = 0; i < dimension; i++) {
        float diff = expectedBits[i] - query[i];
        manualDistance += diff * diff;
    }

    EXPECT_NEAR(computedDistance, manualDistance, 1e-6f);
}

// Test with all zeros and all ones patterns
TEST(ADCFlatCodesDistanceComputerTest, DistanceComputationEdgePatternsTest) {
    int dimension = 16;
    std::vector<float> query = TestHelpers::generateRandomVector(dimension, 0.0f, 1.0f);

    // Test all zeros
    std::vector<uint8_t> allZeros(dimension / 8, 0);
    FaissIndexBQ indexZeros(dimension, allZeros, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computerZeros(
        indexZeros.get_FlatCodesDistanceComputer());
    computerZeros->set_query(query.data());

    float distanceZeros = computerZeros->distance_to_code(allZeros.data());
    float expectedZeros = TestHelpers::computeExpectedL2Distance(query, allZeros);
    EXPECT_NEAR(distanceZeros, expectedZeros, 1e-6f);

    // Test all ones
    std::vector<uint8_t> allOnes(dimension / 8, 0xFF);
    FaissIndexBQ indexOnes(dimension, allOnes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computerOnes(
        indexOnes.get_FlatCodesDistanceComputer());
    computerOnes->set_query(query.data());

    float distanceOnes = computerOnes->distance_to_code(allOnes.data());
    float expectedOnes = TestHelpers::computeExpectedL2Distance(query, allOnes);
    EXPECT_NEAR(distanceOnes, expectedOnes, 1e-6f);
}

// Integration test with multiple vectors - following FaissIndexBQ pattern
TEST(ADCFlatCodesDistanceComputerTest, MultipleVectorsTest) {
    int dimension = 32;
    int numVectors = 5;

    // Create multiple test vectors and pack them into a single codes array
    std::vector<std::vector<float>> testVectors;
    std::vector<uint8_t> allCodes;

    for (int i = 0; i < numVectors; i++) {
        auto vec = TestHelpers::generateTestVector(dimension, i % 4);
        testVectors.push_back(vec);

        auto packed = TestHelpers::packBits(vec);
        allCodes.insert(allCodes.end(), packed.begin(), packed.end());
    }

    std::vector<float> query = TestHelpers::generateRandomVector(dimension);

    // Create FaissIndexBQ with all codes
    FaissIndexBQ index(dimension, allCodes, faiss::METRIC_L2);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        index.get_FlatCodesDistanceComputer());
    computer->set_query(query.data());

    // Test distance computation for each vector
    for (int i = 0; i < numVectors; i++) {
        const uint8_t* codePtr = allCodes.data() + i * (dimension / 8);
        float computedDistance = computer->distance_to_code(codePtr);

        auto packedCode = TestHelpers::packBits(testVectors[i]);
        float expectedDistance = TestHelpers::computeExpectedL2Distance(query, packedCode);

        EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
            << "Vector " << i << " distance mismatch";
    }
}

// Test that demonstrates the actual workflow: index creation -> distance computer -> set query -> compute distances
TEST(ADCFlatCodesDistanceComputerTest, RealWorldWorkflowTest) {
    int dimension = 64;
    int numVectors = 10;

    // Simulate loading codes from a binary index (like in LoadIndexWithStreamADC)
    std::vector<uint8_t> codes;
    std::vector<std::vector<float>> originalVectors;

    // Generate and pack multiple vectors
    for (int i = 0; i < numVectors; i++) {
        auto vec = TestHelpers::generateRandomVector(dimension);
        originalVectors.push_back(vec);

        auto packed = TestHelpers::packBits(vec);
        codes.insert(codes.end(), packed.begin(), packed.end());
    }

    // Create FaissIndexBQ (simulating the alteredStorage in LoadIndexWithStreamADC)
    FaissIndexBQ* alteredStorage = new FaissIndexBQ(dimension, codes, faiss::METRIC_L2);

    // Get distance computer (this is what would be used during search)
    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer(
        alteredStorage->get_FlatCodesDistanceComputer());

    // Set query (this happens during search)
    std::vector<float> query = TestHelpers::generateRandomVector(dimension);
    computer->set_query(query.data());

    // Compute distances to all vectors
    for (int i = 0; i < numVectors; i++) {
        const uint8_t* codePtr = codes.data() + i * (dimension / 8);
        float computedDistance = computer->distance_to_code(codePtr);

        // Verify against expected distance
        auto packedCode = TestHelpers::packBits(originalVectors[i]);
        float expectedDistance = TestHelpers::computeExpectedL2Distance(query, packedCode);

        EXPECT_NEAR(computedDistance, expectedDistance, TestHelpers::NEAR_THRESHOLD)
            << "Distance mismatch for vector " << i;

        // Ensure distance is reasonable (not NaN, not negative for L2)
        EXPECT_FALSE(std::isnan(computedDistance)) << "Distance is NaN for vector " << i;
        EXPECT_GE(computedDistance, 0.0f) << "L2 distance should be non-negative for vector " << i;
    }

    delete alteredStorage;
}

} // namespace faiss_wrapper
} // namespace knn_jni
