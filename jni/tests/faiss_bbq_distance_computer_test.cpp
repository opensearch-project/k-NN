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

#include "sq/faiss_sq_flat.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "gtest/gtest.h"

using idx_t = faiss::idx_t;

namespace knn_jni {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Reference scalar popcount dot-product matching FaissSQDistanceComputer behavior.
// Processes only full 8-byte words (quantizedVectorBytes >> 3), same as the real code.
static uint32_t referencePopcount(const uint8_t* a, const uint8_t* b, int32_t quantizedVectorBytes) {
    const int32_t words = quantizedVectorBytes >> 3;
    uint32_t dp = 0;
    for (int32_t w = 0; w < words; ++w) {
        uint64_t wa, wb;
        std::memcpy(&wa, a + w * 8, sizeof(uint64_t));
        std::memcpy(&wb, b + w * 8, sizeof(uint64_t));
        dp += __builtin_popcountll(wa & wb);
    }
    return dp;
}

// Reference scoring formula matching FaissSQDistanceComputer.
static float referenceScore(bool isMaxIP, int32_t dim, float centroidDp,
                            float ay, float ly, float queryAdditional, float y1,
                            float ax, float lx, float additional, float x1,
                            float dp) {
    float score = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * dp;
    if (isMaxIP) {
        score += queryAdditional + additional - centroidDp;
    } else {
        score = queryAdditional + additional - 2.0f * score;
    }
    return score;
}

// Write correction factors into buffer at ptr.
// Layout: [lower(f32)][upper(f32)][additional(f32)][componentSum(i32)]
static void writeCorrectionFactors(uint8_t* ptr, float lower, float upper,
                                   float additional, int32_t componentSum) {
    std::memcpy(ptr,      &lower,        sizeof(float));
    std::memcpy(ptr + 4,  &upper,        sizeof(float));
    std::memcpy(ptr + 8,  &additional,   sizeof(float));
    std::memcpy(ptr + 12, &componentSum, sizeof(int32_t));
}

// ---------------------------------------------------------------------------
// Test fixture — parameterised over (IsMaxIP, IsBytesMultipleOf8)
// ---------------------------------------------------------------------------

struct SQTestParams {
    bool isMaxIP;
    bool isBytesMultipleOf8;
    std::string name() const {
        std::string s = isMaxIP ? "MaxIP" : "L2";
        s += isBytesMultipleOf8 ? "_Aligned" : "_Unaligned";
        return s;
    }
};

class FaissSQDistanceComputerTest : public ::testing::TestWithParam<SQTestParams> {
protected:
    static constexpr float CENTROID_DP = 0.5f;
    static constexpr float TOLERANCE  = 1e-5f;

    // Deterministic RNG seeded per test for reproducibility.
    std::mt19937 rng{42};

    // Build a data buffer holding `numVecs` elements, each with
    // `quantizedVectorBytes` of random binary data followed by correction factors.
    // Returns (buffer, oneElementSize, quantizedVectorBytes).
    struct TestBuffer {
        // 8-byte aligned storage
        std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> data;
        int32_t oneElementSize;
        int32_t quantizedVectorBytes;
    };

    TestBuffer makeBuffer(int numVecs, int32_t quantizedVectorBytes) {
        const int32_t oneElementSize = quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t);
        TestBuffer buf;
        buf.oneElementSize = oneElementSize;
        buf.quantizedVectorBytes = quantizedVectorBytes;
        buf.data.resize(numVecs * oneElementSize, 0);

        std::uniform_int_distribution<int> byteDist(0, 255);
        std::uniform_real_distribution<float> floatDist(-2.0f, 2.0f);
        std::uniform_int_distribution<int32_t> intDist(-1000, 1000);

        for (int v = 0; v < numVecs; ++v) {
            uint8_t* base = buf.data.data() + v * oneElementSize;
            // Random quantized bytes
            for (int32_t b = 0; b < quantizedVectorBytes; ++b) {
                base[b] = static_cast<uint8_t>(byteDist(rng));
            }
            // Random correction factors
            float lower = floatDist(rng);
            float upper = lower + std::abs(floatDist(rng)) + 0.01f; // ensure upper > lower
            float additional = floatDist(rng);
            int32_t componentSum = intDist(rng);
            writeCorrectionFactors(base + quantizedVectorBytes, lower, upper, additional, componentSum);
        }
        return buf;
    }

    // Create a query buffer (same layout as a single element).
    std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> makeQuery(int32_t quantizedVectorBytes) {
        const int32_t oneElementSize = quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t);
        std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> q(oneElementSize, 0);

        std::uniform_int_distribution<int> byteDist(0, 255);
        std::uniform_real_distribution<float> floatDist(-2.0f, 2.0f);
        std::uniform_int_distribution<int32_t> intDist(-1000, 1000);

        for (int32_t b = 0; b < quantizedVectorBytes; ++b) {
            q[b] = static_cast<uint8_t>(byteDist(rng));
        }
        float lower = floatDist(rng);
        float upper = lower + std::abs(floatDist(rng)) + 0.01f;
        float additional = floatDist(rng);
        int32_t componentSum = intDist(rng);
        writeCorrectionFactors(q.data() + quantizedVectorBytes, lower, upper, additional, componentSum);
        return q;
    }

    // Read correction factors back from a buffer position.
    struct CorrFactors { float lower, interval, additional; float componentSum; };
    static CorrFactors readCorr(const uint8_t* ptr) {
        CorrFactors c;
        float lower, upper;
        std::memcpy(&lower, ptr, sizeof(float));
        std::memcpy(&upper, ptr + 4, sizeof(float));
        std::memcpy(&c.additional, ptr + 8, sizeof(float));
        int32_t cs;
        std::memcpy(&cs, ptr + 12, sizeof(int32_t));
        c.lower = lower;
        c.interval = upper - lower;
        c.componentSum = static_cast<float>(cs);
        return c;
    }

    // Dimension that produces the given quantizedVectorBytes.
    // quantizedVectorBytes = (8 * ((dim+7)/8)) / 32  →  dim ≈ quantizedVectorBytes * 32
    // For aligned (multiple-of-8): use quantizedVectorBytes = 8, 16, 24, ...
    // For unaligned: we artificially test with odd quantizedVectorBytes (e.g. 5, 13).
    int32_t dimForQVB(int32_t qvb) const {
        // Reverse: dim = qvb * 32 (simplification; works for multiples of 8)
        return qvb * 32;
    }
};

// ---------------------------------------------------------------------------
// operator() — single vector distance
// ---------------------------------------------------------------------------

TEST_P(FaissSQDistanceComputerTest, OperatorSingleVector) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    // quantizedVectorBytes is always a multiple of 8 in practice.
    // For IsBytesMultipleOf8=false we still use a multiple-of-8 qvb to exercise
    // the memcpy code path and verify it produces identical results.
    const int32_t qvb = 16;
    const int32_t dim = dimForQVB(qvb);
    constexpr int NUM_VECS = 8;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    // Create distance computer via template
    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));

    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    // Read query correction factors
    auto qCorr = readCorr(queryBuf.data() + qvb);

    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t refDp = referencePopcount(queryBuf.data(), target, qvb);
        auto tCorr = readCorr(target + qvb);

        float expected = referenceScore(isMaxIP, dim, CENTROID_DP,
                                        qCorr.lower, qCorr.interval, qCorr.additional, qCorr.componentSum,
                                        tCorr.lower, tCorr.interval, tCorr.additional, tCorr.componentSum,
                                        static_cast<float>(refDp));

        float actual = (*dc)(static_cast<idx_t>(i));
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "Mismatch at vector " << i << " (isMaxIP=" << isMaxIP
            << ", aligned=" << isBytesMultipleOf8 << ")";
    }
}

// ---------------------------------------------------------------------------
// distances_batch_4
// ---------------------------------------------------------------------------

TEST_P(FaissSQDistanceComputerTest, DistancesBatch4) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    const int32_t qvb = 24;
    const int32_t dim = dimForQVB(qvb);
    constexpr int NUM_VECS = 8;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));

    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    // Batch of 4 — compare against individual operator() calls
    float dis0, dis1, dis2, dis3;
    dc->distances_batch_4(0, 1, 2, 3, dis0, dis1, dis2, dis3);

    EXPECT_NEAR(dis0, (*dc)(0), TOLERANCE);
    EXPECT_NEAR(dis1, (*dc)(1), TOLERANCE);
    EXPECT_NEAR(dis2, (*dc)(2), TOLERANCE);
    EXPECT_NEAR(dis3, (*dc)(3), TOLERANCE);

    // Second batch with different indices
    dc->distances_batch_4(4, 5, 6, 7, dis0, dis1, dis2, dis3);

    EXPECT_NEAR(dis0, (*dc)(4), TOLERANCE);
    EXPECT_NEAR(dis1, (*dc)(5), TOLERANCE);
    EXPECT_NEAR(dis2, (*dc)(6), TOLERANCE);
    EXPECT_NEAR(dis3, (*dc)(7), TOLERANCE);
}

// ---------------------------------------------------------------------------
// symmetric_dis
// ---------------------------------------------------------------------------

TEST_P(FaissSQDistanceComputerTest, SymmetricDis) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    const int32_t qvb = 8;
    const int32_t dim = dimForQVB(qvb);
    constexpr int NUM_VECS = 4;

    auto buf = makeBuffer(NUM_VECS, qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));

    // symmetric_dis doesn't need set_query, it works on stored vectors only
    for (int i = 0; i < NUM_VECS; ++i) {
        for (int j = i; j < NUM_VECS; ++j) {
            const uint8_t* t1 = buf.data.data() + i * buf.oneElementSize;
            const uint8_t* t2 = buf.data.data() + j * buf.oneElementSize;

            uint32_t refDp = referencePopcount(t1, t2, qvb);
            auto c1 = readCorr(t1 + qvb);
            auto c2 = readCorr(t2 + qvb);

            // symmetric scoring formula
            float score = c1.lower * c2.lower * dim
                        + c2.lower * c1.interval * c1.componentSum
                        + c1.lower * c2.interval * c2.componentSum
                        + c1.interval * c2.interval * static_cast<float>(refDp);

            if (isMaxIP) {
                score += c1.additional + c2.additional - CENTROID_DP;
            } else {
                score = c1.additional + c2.additional - 2 * score;
            }

            float actual = dc->symmetric_dis(static_cast<idx_t>(i), static_cast<idx_t>(j));
            EXPECT_NEAR(actual, score, TOLERANCE)
                << "symmetric_dis(" << i << "," << j << ") mismatch";
        }
    }
}

// ---------------------------------------------------------------------------
// setCorrectionFactors — verify extraction matches what was written
// ---------------------------------------------------------------------------

TEST_P(FaissSQDistanceComputerTest, CorrectionFactorsExtraction) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    const int32_t qvb = 16;
    const int32_t dim = dimForQVB(qvb);
    constexpr int NUM_VECS = 4;

    auto buf = makeBuffer(NUM_VECS, qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS));

    // Verify correction factor extraction indirectly: set_query reads query correction
    // factors, then operator() uses both query and target correction factors in the
    // scoring formula. If extraction is wrong, the score won't match the reference.
    auto queryBuf = makeQuery(qvb);
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    auto qCorr = readCorr(queryBuf.data() + qvb);

    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t refDp = referencePopcount(queryBuf.data(), target, qvb);
        auto tCorr = readCorr(target + qvb);

        float expected = referenceScore(isMaxIP, dim, CENTROID_DP,
                                        qCorr.lower, qCorr.interval, qCorr.additional, qCorr.componentSum,
                                        tCorr.lower, tCorr.interval, tCorr.additional, tCorr.componentSum,
                                        static_cast<float>(refDp));

        float actual = (*dc)(static_cast<idx_t>(i));
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "Correction factor extraction mismatch at vector " << i;
    }
}

// ---------------------------------------------------------------------------
// FaissSQFlat::get_distance_computer — integration test
// ---------------------------------------------------------------------------

TEST_P(FaissSQDistanceComputerTest, GetDistanceComputerIntegration) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    const int32_t qvb = 16;
    const int32_t dim = dimForQVB(qvb);
    constexpr int NUM_VECS = 4;

    faiss::MetricType metric = isMaxIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    FaissSQFlat flat(NUM_VECS, qvb, CENTROID_DP, dim, metric);

    // Populate the storage
    auto buf = makeBuffer(NUM_VECS, qvb);
    flat.quantizedVectorsAndCorrectionFactors.assign(buf.data.begin(), buf.data.end());
    flat.ntotal = NUM_VECS;

    // get_distance_computer should pick the right template
    std::unique_ptr<faiss::DistanceComputer> dc(flat.get_distance_computer());

    auto queryBuf = makeQuery(qvb);
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    auto qCorr = readCorr(queryBuf.data() + qvb);

    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t refDp = referencePopcount(queryBuf.data(), target, qvb);
        auto tCorr = readCorr(target + qvb);

        float expected = referenceScore(isMaxIP, dim, CENTROID_DP,
                                        qCorr.lower, qCorr.interval, qCorr.additional, qCorr.componentSum,
                                        tCorr.lower, tCorr.interval, tCorr.additional, tCorr.componentSum,
                                        static_cast<float>(refDp));

        float actual = (*dc)(static_cast<idx_t>(i));
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "Integration mismatch at vector " << i;
    }
}

// ---------------------------------------------------------------------------
// Instantiate all 4 combinations
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    SQDistanceComputer,
    FaissSQDistanceComputerTest,
    ::testing::Values(
        SQTestParams{false, true},   // L2, aligned
        SQTestParams{false, false},  // L2, unaligned
        SQTestParams{true,  true},   // MaxIP, aligned
        SQTestParams{true,  false}   // MaxIP, unaligned
    ),
    [](const ::testing::TestParamInfo<SQTestParams>& info) {
        return info.param.name();
    }
);

} // namespace knn_jni
