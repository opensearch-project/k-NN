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
        // Negate: Faiss HNSW always minimizes distance (CMax comparator).
        return -score;
    } else {
        return queryAdditional + additional - 2.0f * score;
    }
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
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

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
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

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
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

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
                score = -score;
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
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (isMaxIP && !isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else if (!isMaxIP && isBytesMultipleOf8)
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

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
    FaissSQFlat flat(NUM_VECS, qvb, CENTROID_DP, dim, metric, 1);

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
// Non-multiple-of-8 quantizedVectorBytes (e.g. dim=56 → 7 bytes)
// Verifies the byte remainder loop handles trailing bytes correctly.
// ---------------------------------------------------------------------------

// Reference popcount that handles remainder bytes (matches the fixed code).
static uint32_t referencePopcountWithRemainder(const uint8_t* a, const uint8_t* b, int32_t quantizedVectorBytes) {
    const int32_t words = quantizedVectorBytes >> 3;
    uint32_t dp = 0;
    for (int32_t w = 0; w < words; ++w) {
        uint64_t wa, wb;
        std::memcpy(&wa, a + w * 8, sizeof(uint64_t));
        std::memcpy(&wb, b + w * 8, sizeof(uint64_t));
        dp += __builtin_popcountll(wa & wb);
    }
    const int32_t remainStart = words * 8;
    for (int32_t r = remainStart; r < quantizedVectorBytes; ++r) {
        dp += __builtin_popcount((a[r] & b[r]) & 0xFF);
    }
    return dp;
}

TEST_P(FaissSQDistanceComputerTest, OperatorNonMultipleOf8Bytes) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    // Only the unaligned path handles remainder bytes; skip aligned tests.
    if (isBytesMultipleOf8) return;

    // dim=56 → quantizedVectorBytes=7 (the exact case that triggered the bug)
    const int32_t qvb = 7;
    const int32_t dim = 56;
    constexpr int NUM_VECS = 8;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    auto qCorr = readCorr(queryBuf.data() + qvb);

    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t refDp = referencePopcountWithRemainder(queryBuf.data(), target, qvb);
        auto tCorr = readCorr(target + qvb);

        float expected = referenceScore(isMaxIP, dim, CENTROID_DP,
                                        qCorr.lower, qCorr.interval, qCorr.additional, qCorr.componentSum,
                                        tCorr.lower, tCorr.interval, tCorr.additional, tCorr.componentSum,
                                        static_cast<float>(refDp));

        float actual = (*dc)(static_cast<idx_t>(i));
        // Verify the dot product is non-zero (the original bug produced dp=0)
        EXPECT_NE(refDp, 0u) << "Reference dp should be non-zero for random data at vector " << i;
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "Mismatch at vector " << i << " with qvb=" << qvb;
    }
}

TEST_P(FaissSQDistanceComputerTest, DistancesBatch4NonMultipleOf8Bytes) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    if (isBytesMultipleOf8) return;

    const int32_t qvb = 7;
    const int32_t dim = 56;
    constexpr int NUM_VECS = 8;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    float dis0, dis1, dis2, dis3;
    dc->distances_batch_4(0, 1, 2, 3, dis0, dis1, dis2, dis3);

    EXPECT_NEAR(dis0, (*dc)(0), TOLERANCE);
    EXPECT_NEAR(dis1, (*dc)(1), TOLERANCE);
    EXPECT_NEAR(dis2, (*dc)(2), TOLERANCE);
    EXPECT_NEAR(dis3, (*dc)(3), TOLERANCE);
}

TEST_P(FaissSQDistanceComputerTest, SymmetricDisNonMultipleOf8Bytes) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    if (isBytesMultipleOf8) return;

    const int32_t qvb = 7;
    const int32_t dim = 56;
    constexpr int NUM_VECS = 4;

    auto buf = makeBuffer(NUM_VECS, qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP)
        dc.reset(new FaissSQDistanceComputer<true, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    else
        dc.reset(new FaissSQDistanceComputer<false, false>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));

    for (int i = 0; i < NUM_VECS; ++i) {
        for (int j = i; j < NUM_VECS; ++j) {
            const uint8_t* t1 = buf.data.data() + i * buf.oneElementSize;
            const uint8_t* t2 = buf.data.data() + j * buf.oneElementSize;

            uint32_t refDp = referencePopcountWithRemainder(t1, t2, qvb);
            auto c1 = readCorr(t1 + qvb);
            auto c2 = readCorr(t2 + qvb);

            float score = c1.lower * c2.lower * dim
                        + c2.lower * c1.interval * c1.componentSum
                        + c1.lower * c2.interval * c2.componentSum
                        + c1.interval * c2.interval * static_cast<float>(refDp);

            if (isMaxIP) {
                score += c1.additional + c2.additional - CENTROID_DP;
                score = -score;
            } else {
                score = c1.additional + c2.additional - 2 * score;
            }

            float actual = dc->symmetric_dis(static_cast<idx_t>(i), static_cast<idx_t>(j));
            EXPECT_NEAR(actual, score, TOLERANCE)
                << "symmetric_dis(" << i << "," << j << ") mismatch with qvb=" << qvb;
        }
    }
}

TEST_P(FaissSQDistanceComputerTest, GetDistanceComputerIntegrationNonMultipleOf8) {
    auto [isMaxIP, isBytesMultipleOf8] = GetParam();
    // This test exercises FaissSQFlat::get_distance_computer with non-aligned element size.
    // Only run once per metric (skip the aligned param to avoid duplicate).
    if (isBytesMultipleOf8) return;

    const int32_t qvb = 7;  // dim=56
    const int32_t dim = 56;
    constexpr int NUM_VECS = 4;

    faiss::MetricType metric = isMaxIP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    FaissSQFlat flat(NUM_VECS, qvb, CENTROID_DP, dim, metric, 1);

    auto buf = makeBuffer(NUM_VECS, qvb);
    flat.quantizedVectorsAndCorrectionFactors.assign(buf.data.begin(), buf.data.end());
    flat.ntotal = NUM_VECS;

    // get_distance_computer should pick IsBytesMultipleOf8=false since oneElementSize=23
    std::unique_ptr<faiss::DistanceComputer> dc(flat.get_distance_computer());

    auto queryBuf = makeQuery(qvb);
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    auto qCorr = readCorr(queryBuf.data() + qvb);

    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t refDp = referencePopcountWithRemainder(queryBuf.data(), target, qvb);
        auto tCorr = readCorr(target + qvb);

        float expected = referenceScore(isMaxIP, dim, CENTROID_DP,
                                        qCorr.lower, qCorr.interval, qCorr.additional, qCorr.componentSum,
                                        tCorr.lower, tCorr.interval, tCorr.additional, tCorr.componentSum,
                                        static_cast<float>(refDp));

        float actual = (*dc)(static_cast<idx_t>(i));
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "Integration mismatch at vector " << i << " with qvb=" << qvb;
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

// ---------------------------------------------------------------------------
// Multi-bit tests (docBits in {1, 2, 4})
// ---------------------------------------------------------------------------
//
// The tests above all pass docBits=1. These new tests cover B=2 and B=4 by
// comparing FaissSQDistanceComputer output against a scalar reference that
// mirrors the production formula:
//   dp = Σ_{i,j<docBits} popcount(planeA_i AND planeB_j) << (i + j)
// and validates that the intervalLength scaling by 1/(2^docBits - 1) applies
// correctly (no-op for B=1, 1/3 for B=2, 1/15 for B=4).

struct MultiBitParams {
    bool isMaxIP;
    int32_t docBits;
    std::string name() const {
        return std::string(isMaxIP ? "MaxIP" : "L2") + "_B" + std::to_string(docBits);
    }
};

class FaissSQDistanceComputerMultiBitTest : public ::testing::TestWithParam<MultiBitParams> {
protected:
    static constexpr float CENTROID_DP = 0.5f;
    static constexpr float TOLERANCE  = 1e-4f;

    std::mt19937 rng{123};

    struct Buffer {
        std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> data;
        int32_t oneElementSize;
        int32_t quantizedVectorBytes;
    };

    // qvb must be a multiple of docBits so `planeBytes = qvb / docBits` is exact.
    // Aligned qvb (multiple of 8) also exercises the fast reinterpret_cast path.
    Buffer makeBuffer(int numVecs, int32_t quantizedVectorBytes) {
        const int32_t oneElementSize = quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t);
        Buffer buf;
        buf.oneElementSize = oneElementSize;
        buf.quantizedVectorBytes = quantizedVectorBytes;
        buf.data.resize(numVecs * oneElementSize, 0);

        std::uniform_int_distribution<int> byteDist(0, 255);
        std::uniform_real_distribution<float> floatDist(-2.0f, 2.0f);
        std::uniform_int_distribution<int32_t> intDist(-1000, 1000);

        for (int v = 0; v < numVecs; ++v) {
            uint8_t* base = buf.data.data() + v * oneElementSize;
            for (int32_t b = 0; b < quantizedVectorBytes; ++b) {
                base[b] = static_cast<uint8_t>(byteDist(rng));
            }
            float lower = floatDist(rng);
            float upper = lower + std::abs(floatDist(rng)) + 0.01f;
            float additional = floatDist(rng);
            int32_t componentSum = intDist(rng);
            writeCorrectionFactors(base + quantizedVectorBytes, lower, upper, additional, componentSum);
        }
        return buf;
    }

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

    // Reference multiBitDp. The formula depends on the docBits Lucene layout:
    //   B=1, B=2 (bit-plane popcount): Σ_{i,j} popcount(planeA_i AND planeB_j) << (i + j)
    //   B=4 (PACKED_NIBBLE):           Σ_i (aHi_i * bHi_i + aLo_i * bLo_i)
    // We dispatch on docBits so this reference matches production for every supported width.
    static uint64_t referenceMultiBitDp(const uint8_t* a, const uint8_t* b, int32_t quantizedVectorBytes, int32_t docBits) {
        if (docBits == 4) {
            // PACKED_NIBBLE byte-wise multiply: each byte holds two 4-bit elements.
            uint64_t total = 0;
            for (int32_t k = 0; k < quantizedVectorBytes; ++k) {
                const uint32_t aLo = a[k] & 0x0Fu;
                const uint32_t aHi = (a[k] >> 4) & 0x0Fu;
                const uint32_t bLo = b[k] & 0x0Fu;
                const uint32_t bHi = (b[k] >> 4) & 0x0Fu;
                total += aLo * bLo + aHi * bHi;
            }
            return total;
        }
        // B=1, B=2: bit-plane popcount-AND-shift double sum.
        const uint64_t planeBytes = static_cast<uint64_t>(quantizedVectorBytes) / static_cast<uint64_t>(docBits);
        uint64_t dp = 0;
        for (int32_t i = 0; i < docBits; ++i) {
            const uint8_t* pa = a + static_cast<uint64_t>(i) * planeBytes;
            for (int32_t j = 0; j < docBits; ++j) {
                const uint8_t* pb = b + static_cast<uint64_t>(j) * planeBytes;
                uint64_t pc = 0;
                for (uint64_t k = 0; k < planeBytes; ++k) {
                    pc += __builtin_popcount((pa[k] & pb[k]) & 0xFF);
                }
                dp += pc << (i + j);
            }
        }
        return dp;
    }

    // Reference score using the docBits-scaled intervalLength.
    static float referenceScoreMultiBit(bool isMaxIP, int32_t dim, float centroidDp,
                                         float lowerY, float rawIntervalY, float additionalY, float y1,
                                         float lowerX, float rawIntervalX, float additionalX, float x1,
                                         uint64_t dp, int32_t docBits) {
        const float scale = 1.0f / static_cast<float>((1 << docBits) - 1);
        const float ay = lowerY;
        const float ly = rawIntervalY * scale;
        const float ax = lowerX;
        const float lx = rawIntervalX * scale;
        float score = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * static_cast<float>(dp);
        if (isMaxIP) {
            score += additionalY + additionalX - centroidDp;
            return -score;
        } else {
            return additionalY + additionalX - 2.0f * score;
        }
    }

    struct RawCorr { float lower, rawInterval, additional; float componentSum; };
    static RawCorr readRawCorr(const uint8_t* ptr) {
        RawCorr c;
        float lower, upper;
        std::memcpy(&lower, ptr, sizeof(float));
        std::memcpy(&upper, ptr + 4, sizeof(float));
        std::memcpy(&c.additional, ptr + 8, sizeof(float));
        int32_t cs;
        std::memcpy(&cs, ptr + 12, sizeof(int32_t));
        c.lower = lower;
        c.rawInterval = upper - lower;
        c.componentSum = static_cast<float>(cs);
        return c;
    }
};

// Operator() bit-identical to scalar reference across docBits ∈ {1, 2, 4}.
TEST_P(FaissSQDistanceComputerMultiBitTest, OperatorMatchesReference) {
    auto [isMaxIP, docBits] = GetParam();
    // planeBytes must divide quantizedVectorBytes cleanly. Pick qvb = docBits * 16 (multiple of 8 → aligned path).
    const int32_t qvb = docBits * 16;
    const int32_t planeBytes = qvb / docBits;
    // dim doesn't matter for correctness beyond feeding the score formula; use planeBytes*8.
    const int32_t dim = planeBytes * 8;
    constexpr int NUM_VECS = 8;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    // Use the aligned template path (qvb is a multiple of 8 by construction here).
    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP) {
        dc.reset(new FaissSQDistanceComputer<true,  true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    } else {
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    }
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    auto qCorr = readRawCorr(queryBuf.data() + qvb);
    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        const uint64_t refDp = referenceMultiBitDp(queryBuf.data(), target, qvb, docBits);
        auto tCorr = readRawCorr(target + qvb);
        float expected = referenceScoreMultiBit(
            isMaxIP, dim, CENTROID_DP,
            qCorr.lower, qCorr.rawInterval, qCorr.additional, qCorr.componentSum,
            tCorr.lower, tCorr.rawInterval, tCorr.additional, tCorr.componentSum,
            refDp, docBits);
        float actual = (*dc)(static_cast<idx_t>(i));
        EXPECT_NEAR(actual, expected, TOLERANCE)
            << "docBits=" << docBits << ", isMaxIP=" << isMaxIP << ", i=" << i;
    }
}

// symmetric_dis (build-time distance between two stored codes) bit-identical to reference.
TEST_P(FaissSQDistanceComputerMultiBitTest, SymmetricDisMatchesReference) {
    auto [isMaxIP, docBits] = GetParam();
    const int32_t qvb = docBits * 16;
    const int32_t planeBytes = qvb / docBits;
    const int32_t dim = planeBytes * 8;
    constexpr int NUM_VECS = 6;

    auto buf = makeBuffer(NUM_VECS, qvb);

    std::unique_ptr<FaissSQDistanceComputer<true, true>> dcMax;
    std::unique_ptr<FaissSQDistanceComputer<false, true>> dcL2;
    // We need the concrete type to call symmetric_dis (which isn't a virtual).
    if (isMaxIP) {
        dcMax.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    } else {
        dcL2.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    }

    for (int i = 0; i < NUM_VECS; ++i) {
        for (int j = i; j < NUM_VECS; ++j) {
            const uint8_t* a = buf.data.data() + i * buf.oneElementSize;
            const uint8_t* b = buf.data.data() + j * buf.oneElementSize;
            const uint64_t refDp = referenceMultiBitDp(a, b, qvb, docBits);
            auto aCorr = readRawCorr(a + qvb);
            auto bCorr = readRawCorr(b + qvb);

            // symmetric_dis mirrors scoringSecondPart structure but with both sides scaled.
            const float scale = 1.0f / static_cast<float>((1 << docBits) - 1);
            const float ax = aCorr.lower;
            const float lx = aCorr.rawInterval * scale;
            const float az = bCorr.lower;
            const float lz = bCorr.rawInterval * scale;
            float score = ax * az * dim + az * lx * aCorr.componentSum + ax * lz * bCorr.componentSum
                        + lx * lz * static_cast<float>(refDp);
            float expected;
            if (isMaxIP) {
                score += aCorr.additional + bCorr.additional - CENTROID_DP;
                expected = -score;
            } else {
                expected = aCorr.additional + bCorr.additional - 2.0f * score;
            }

            float actual = isMaxIP ? dcMax->symmetric_dis(i, j) : dcL2->symmetric_dis(i, j);
            EXPECT_NEAR(actual, expected, TOLERANCE)
                << "docBits=" << docBits << ", isMaxIP=" << isMaxIP << ", (i,j)=(" << i << "," << j << ")";
        }
    }
}

// distances_batch_4 must agree with per-element operator() at each docBits.
TEST_P(FaissSQDistanceComputerMultiBitTest, DistancesBatch4MatchesSingle) {
    auto [isMaxIP, docBits] = GetParam();
    const int32_t qvb = docBits * 16;
    const int32_t planeBytes = qvb / docBits;
    const int32_t dim = planeBytes * 8;
    constexpr int NUM_VECS = 6;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP) {
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    } else {
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, docBits));
    }
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    float d0, d1, d2, d3;
    dc->distances_batch_4(0, 1, 2, 3, d0, d1, d2, d3);
    EXPECT_NEAR(d0, (*dc)(0), TOLERANCE) << "docBits=" << docBits;
    EXPECT_NEAR(d1, (*dc)(1), TOLERANCE) << "docBits=" << docBits;
    EXPECT_NEAR(d2, (*dc)(2), TOLERANCE) << "docBits=" << docBits;
    EXPECT_NEAR(d3, (*dc)(3), TOLERANCE) << "docBits=" << docBits;
}

// docBits=1 must produce results bit-identical to the legacy single-bit path,
// which is the "no regression on existing 1-bit graphs" invariant the POC commit relies on.
TEST_P(FaissSQDistanceComputerMultiBitTest, DocBits1MatchesLegacyPopcount) {
    auto [isMaxIP, docBits] = GetParam();
    if (docBits != 1) {
        GTEST_SKIP() << "This invariant only applies to docBits=1";
    }
    const int32_t qvb = 24;
    const int32_t dim = qvb * 32;
    constexpr int NUM_VECS = 4;

    auto buf = makeBuffer(NUM_VECS, qvb);
    auto queryBuf = makeQuery(qvb);

    std::unique_ptr<faiss::DistanceComputer> dc;
    if (isMaxIP) {
        dc.reset(new FaissSQDistanceComputer<true, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    } else {
        dc.reset(new FaissSQDistanceComputer<false, true>(buf.oneElementSize, buf.data.data(), CENTROID_DP, dim, NUM_VECS, 1));
    }
    dc->set_query(reinterpret_cast<const float*>(queryBuf.data()));

    // For docBits=1, dataScale = 1.0 so intervalLength is unchanged, and multiBitDp reduces
    // to popcount(a AND b). So the legacy referencePopcount + referenceScore path (used in
    // the SQTestParams tests above) must agree with the multi-bit reference too.
    for (int i = 0; i < NUM_VECS; ++i) {
        const uint8_t* target = buf.data.data() + i * buf.oneElementSize;
        uint32_t legacyDp = referencePopcount(queryBuf.data(), target, qvb);
        uint64_t multiBitRef = referenceMultiBitDp(queryBuf.data(), target, qvb, 1);
        EXPECT_EQ(legacyDp, multiBitRef)
            << "docBits=1 multi-bit dp must equal legacy popcount, i=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultiBit,
    FaissSQDistanceComputerMultiBitTest,
    ::testing::Values(
        MultiBitParams{false, 1}, MultiBitParams{false, 2}, MultiBitParams{false, 4},
        MultiBitParams{true,  1}, MultiBitParams{true,  2}, MultiBitParams{true,  4}
    ),
    [](const ::testing::TestParamInfo<MultiBitParams>& info) { return info.param.name(); }
);

// ---------------------------------------------------------------------------
// docBits validation — constructor must reject unsupported / dangerous values
// ---------------------------------------------------------------------------

TEST(FaissSQDistanceComputerValidation, RejectsUnsupportedDocBits) {
    // We only need a dummy buffer big enough to compute oneElementByteSize; nothing is dereferenced
    // because the constructor validates docBits before touching any data.
    // oneElementByteSize must be > 16 (correction factors) for quantizedVectorBytes to be positive.
    constexpr int32_t qvb = 8;
    constexpr int32_t oneElementByteSize = qvb + 3 * sizeof(float) + sizeof(int32_t);
    std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> dummy(oneElementByteSize, 0);

    // docBits=0 would trigger div-by-zero in planeBytes computation.
    EXPECT_THROW(
        (FaissSQDistanceComputer<false, true>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/0)),
        std::runtime_error);
    // docBits=3 is not a supported MOS width.
    EXPECT_THROW(
        (FaissSQDistanceComputer<false, true>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/3)),
        std::runtime_error);
    // docBits=8 exceeds Lucene's MOS bit widths.
    EXPECT_THROW(
        (FaissSQDistanceComputer<false, true>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/8)),
        std::runtime_error);
    // Negative docBits — the (1 << bits) computation would be UB.
    EXPECT_THROW(
        (FaissSQDistanceComputer<false, true>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/-1)),
        std::runtime_error);
}

TEST(FaissSQDistanceComputerValidation, RejectsOddQuantizedVectorBytesAtB2) {
    // qvb=9 is not divisible by docBits=2. B=2 requires two bit planes of equal length,
    // so quantizedVectorBytes must be even. B=1 and B=4 have no such requirement.
    constexpr int32_t qvb = 9;
    constexpr int32_t oneElementByteSize = qvb + 3 * sizeof(float) + sizeof(int32_t);
    std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> dummy(oneElementByteSize, 0);

    // B=2 must throw because 9 is odd.
    EXPECT_THROW(
        (FaissSQDistanceComputer<false, false>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/2)),
        std::runtime_error);

    // B=1 is fine (any qvb is valid — planeBytes == qvb).
    EXPECT_NO_THROW(
        (FaissSQDistanceComputer<false, false>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/1)));

    // B=4 is fine too — PACKED_NIBBLE bytes are independent (each byte holds 2 elements),
    // so qvb=9 doesn't need any divisibility property beyond "positive".
    EXPECT_NO_THROW(
        (FaissSQDistanceComputer<false, false>(oneElementByteSize, dummy.data(), 0.0f, 8, 1, /*docBits=*/4)));
}

TEST(FaissSQDistanceComputerValidation, AcceptsAllSupportedDocBits) {
    // Verify the happy path for each supported docBits width. qvb must be a multiple of docBits.
    for (int32_t docBits : {1, 2, 4}) {
        const int32_t qvb = docBits * 16; // multiple of docBits AND multiple of 8
        const int32_t oneElementByteSize = qvb + 3 * sizeof(float) + sizeof(int32_t);
        std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> dummy(oneElementByteSize, 0);

        EXPECT_NO_THROW(
            (FaissSQDistanceComputer<false, true>(oneElementByteSize, dummy.data(), 0.0f, 128, 1, docBits))
        ) << "docBits=" << docBits << " must be accepted";
    }
}

} // namespace knn_jni
