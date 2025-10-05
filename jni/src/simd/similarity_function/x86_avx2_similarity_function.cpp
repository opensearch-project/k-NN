#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <iostream>
#include <cstdint>

#define __NO_SELECT_FUNCTION
#include "default_simd_similarity_function.cpp"
#undef __NO_SELECT_FUNCTION
#include "parameter_utils.h"
#include "memory_util.h"



//
// FP16 Inner product
//

struct FP16InnerProductSimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   int32_t numVectors) final {
        // Get search context
        uint8_t* vectors[8];
        const int32_t dimension = srchContext->dimension;

        // Bulk SIMD with batch size 8
        int32_t i = 0;
        for ( ; (i + 8) <= numVectors ; i += 8, scores += 8) {
            srchContext->getVectorPointersInBulk(vectors, internalVectorIds + i, 8);
            batchInnerProduct8FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                vectors[4], vectors[5], vectors[6], vectors[7],
                dimension,
                scores[0], scores[1], scores[2], scores[3],
                scores[4], scores[5], scores[6], scores[7]);
        }

        // Bulk SIMD with batch size 4
        for ( ; (i + 4) <= numVectors ; i += 4, scores += 4) {
            srchContext->getVectorPointersInBulk(vectors, internalVectorIds + i, 4);
            batchInnerProduct4FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                dimension,
                scores[0], scores[1], scores[2], scores[3]);
        }

        // Transform score value to MAX_IP
        for (int32_t j = 0 ; j < i ; ++j) {
            scores[j] = scores[j] < 0 ? (1 / (1 - scores[j])) : (scores[j] + 1);
        }
        FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, i);

        // Handle remaining vectors
        while (i < numVectors) {
            *scores++ = calculateSimilarity(srchContext, internalVectorIds[i++]);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) final {
        return DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC.calculateSimilarity(srchContext, internalVectorId);
    }

private:
    static void batchInnerProduct8FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0, const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2, const uint8_t* RESTRICT d3,
        const uint8_t* RESTRICT d4, const uint8_t* RESTRICT d5,
        const uint8_t* RESTRICT d6, const uint8_t* RESTRICT d7,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3,
        float& score4, float& score5, float& score6, float& score7) {

        // Accumulators (8 vectors, one for each target)
        // Initialize all 8 FP32 elements in the __m256 registers to 0.0f
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        // The loop processes 8 elements (256 bits) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Using _mm256_load_ps for guaranteed 64-byte aligned query vector
            __m256 q = _mm256_load_ps(query + i);

            // --- 2. Load and Convert Target Vectors (8 targets * 8 FP16 values each) ---
            // Use F16C's _mm256_cvtph_ps to convert 8 FP16 values (128 bits)
            // into 8 FP32 values (256 bits).
            // Target loads use _mm_loadu_si128 to load 128 bits (8 FP16) unaligned.
            __m256 d0_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d0 + i * 2)));
            __m256 d1_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d1 + i * 2)));
            __m256 d2_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d2 + i * 2)));
            __m256 d3_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d3 + i * 2)));
            __m256 d4_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d4 + i * 2)));
            __m256 d5_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d5 + i * 2)));
            __m256 d6_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d6 + i * 2)));
            __m256 d7_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d7 + i * 2)));

            // --- 3. Compute Product and Accumulate ---
            // Inner Product Step: acc = (Q * D) + acc using FMA

            acc0 = _mm256_fmadd_ps(q, d0_vec, acc0);
            acc1 = _mm256_fmadd_ps(q, d1_vec, acc1);
            acc2 = _mm256_fmadd_ps(q, d2_vec, acc2);
            acc3 = _mm256_fmadd_ps(q, d3_vec, acc3);
            acc4 = _mm256_fmadd_ps(q, d4_vec, acc4);
            acc5 = _mm256_fmadd_ps(q, d5_vec, acc5);
            acc6 = _mm256_fmadd_ps(q, d6_vec, acc6);
            acc7 = _mm256_fmadd_ps(q, d7_vec, acc7);
        }

        // --- 4. Final Reduction (Horizontal Sum) ---
        auto horizontal_sum_avx2 = [](__m256 v) -> float {
            // Step 1: Horizontal add
            __m256 hsum = _mm256_hadd_ps(v, v);

            // Step 2: Second horizontal add
            hsum = _mm256_hadd_ps(hsum, hsum);

            // Step 3: Extract the 128-bit halves and add them (V0..V3 + V4..V7)
            __m128 hsum_128 = _mm256_extractf128_ps(hsum, 1); // Get upper 128 bits
            __m128 hsum_rest = _mm256_castps256_ps128(hsum); // Get lower 128 bits

            // Step 4: Add the two 128-bit sums together
            __m128 final_sum = _mm_add_ss(hsum_128, hsum_rest);

            // Step 5: Extract the result
            return _mm_cvtss_f32(final_sum);
        };

        score0 = horizontal_sum_avx2(acc0);
        score1 = horizontal_sum_avx2(acc1);
        score2 = horizontal_sum_avx2(acc2);
        score3 = horizontal_sum_avx2(acc3);
        score4 = horizontal_sum_avx2(acc4);
        score5 = horizontal_sum_avx2(acc5);
        score6 = horizontal_sum_avx2(acc6);
        score7 = horizontal_sum_avx2(acc7);

        // --- 5. Scalar Tail ---
        // Handle the remaining elements (dim % 8)
        for (; i < dim; i++) {
            const float qv = query[i];

            // Load and convert target vector elements
            float tv0 = (float)*((const _Float16 *)(d0 + i * 2));
            float tv1 = (float)*((const _Float16 *)(d1 + i * 2));
            float tv2 = (float)*((const _Float16 *)(d2 + i * 2));
            float tv3 = (float)*((const _Float16 *)(d3 + i * 2));
            float tv4 = (float)*((const _Float16 *)(d4 + i * 2));
            float tv5 = (float)*((const _Float16 *)(d5 + i * 2));
            float tv6 = (float)*((const _Float16 *)(d6 + i * 2));
            float tv7 = (float)*((const _Float16 *)(d7 + i * 2));

            // Accumulate product
            score0 += qv * tv0;
            score1 += qv * tv1;
            score2 += qv * tv2;
            score3 += qv * tv3;
            score4 += qv * tv4;
            score5 += qv * tv5;
            score6 += qv * tv6;
            score7 += qv * tv7;
        }
    }

    void batchInnerProduct4FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0,
        const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2,
        const uint8_t* RESTRICT d3,
        size_t dim,
        float& score0,
        float& score1,
        float& score2,
        float& score3) {

        // Local lambda for horizontal sum of __m256 (8 floats) for AVX2.
        // This replaces the static inline helper function.
        auto hsum = [](__m256 v) -> float {
            // 1. Sum the lower and upper 128-bit lanes
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            __m128 vsum = _mm_add_ps(vlow, vhigh); // vsum now holds 4 elements of the full sum

            // 2. Perform horizontal add on the resulting __m128 (sum of 4 elements)
            vsum = _mm_hadd_ps(vsum, vsum);
            vsum = _mm_hadd_ps(vsum, vsum); // Sums all 4 elements into the lowest position
            return _mm_cvtss_f32(vsum);
        };

        // Accumulators (4 vectors, one for each target)
        // Initialize all 8 FP32 elements in the __m256 registers to 0.0f
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // The loop processes 8 elements (256 bits) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Using _mm256_load_ps for guaranteed 64-byte aligned query vector
            __m256 q = _mm256_load_ps(query + i);

            // --- 2. Load and Convert Target Vectors (4 targets * 8 FP16 values each) ---
            // Use F16C's _mm256_cvtph_ps to convert 8 FP16 values (128 bits)
            // into 8 FP32 values (256 bits).
            // Target loads use _mm_loadu_si128 to load 128 bits (8 FP16) unaligned.
            __m256 d0_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d0 + i * 2)));
            __m256 d1_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d1 + i * 2)));
            __m256 d2_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d2 + i * 2)));
            __m256 d3_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d3 + i * 2)));

            // --- 3. Compute Product and Accumulate ---
            // Inner Product Step: acc = (Q * D) + acc using FMA

            acc0 = _mm256_fmadd_ps(q, d0_vec, acc0);
            acc1 = _mm256_fmadd_ps(q, d1_vec, acc1);
            acc2 = _mm256_fmadd_ps(q, d2_vec, acc2);
            acc3 = _mm256_fmadd_ps(q, d3_vec, acc3);
        }

        // --- 4. Final Reduction (Horizontal Sum) ---
        score0 = hsum(acc0);
        score1 = hsum(acc1);
        score2 = hsum(acc2);
        score3 = hsum(acc3);

        // --- 5. Scalar Tail ---
        // Handle the remaining elements (dim % 8)
        for (; i < dim; i++) {
            const float qv = query[i];

            // Load and convert target vector elements
            float tv0 = (float)*((const _Float16 *)(d0 + i * 2));
            float tv1 = (float)*((const _Float16 *)(d1 + i * 2));
            float tv2 = (float)*((const _Float16 *)(d2 + i * 2));
            float tv3 = (float)*((const _Float16 *)(d3 + i * 2));

            // Accumulate product
            score0 += qv * tv0;
            score1 += qv * tv1;
            score2 += qv * tv2;
            score3 += qv * tv3;
        }
    }
};

FP16InnerProductSimilarityFunction FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;

//
// FP16 L2
//

struct FP16L2SimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   int32_t numVectors) final {
        // Get search context
        uint8_t* vectors[8];
        const int32_t dimension = srchContext->dimension;

        // Bulk SIMD with batch size 8
        int32_t i = 0;
        for ( ; (i + 8) <= numVectors ; i += 8, scores += 8) {
            srchContext->getVectorPointersInBulk(vectors, internalVectorIds + i, 8);
            batchEuclidian8FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                vectors[4], vectors[5], vectors[6], vectors[7],
                dimension,
                scores[0], scores[1], scores[2], scores[3],
                scores[4], scores[5], scores[6], scores[7]);
        }

        // Bulk SIMD with batch size 4
        for ( ; (i + 4) <= numVectors ; i += 4, scores += 4) {
            srchContext->getVectorPointersInBulk(vectors, internalVectorIds + i, 4);
            batchEuclidian4FP16Targets(
                reinterpret_cast<float*>(srchContext->queryVectorSimdAligned),
                vectors[0], vectors[1], vectors[2], vectors[3],
                dimension,
                scores[0], scores[1], scores[2], scores[3]);
        }

        // Transform scores
        FaissScoreToLuceneScoreTransform::l2TransformBulk(scores, i);

        // Handle remaining vectors
        while (i < numVectors) {
            *scores++ = calculateSimilarity(srchContext, internalVectorIds[i++]);
        }
   }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) final {
        return DEFAULT_FP16_L2_SIMIL_FUNC.calculateSimilarity(srchContext, internalVectorId);
    }

private:
    static void batchEuclidian8FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0, const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2, const uint8_t* RESTRICT d3,
        const uint8_t* RESTRICT d4, const uint8_t* RESTRICT d5,
        const uint8_t* RESTRICT d6, const uint8_t* RESTRICT d7,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3,
        float& score4, float& score5, float& score6, float& score7) {

        // Accumulators (8 vectors, one for each target)
        // Initialize all 8 FP32 elements in the __m256 registers to 0.0f
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        // The loop processes 8 elements (256 bits) at a time.
        const size_t dim_aligned = dim & ~7;
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            __m256 q = _mm256_loadu_ps(query + i);

            // --- 2. Load and Convert Target Vectors (8 targets * 8 FP16 values each) ---
            // Use F16C's _mm256_cvtph_ps to convert 8 FP16 values (128 bits)
            // into 8 FP32 values (256 bits) in one instruction.

            __m256 d0_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d0 + i * 2)));
            __m256 d1_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d1 + i * 2)));
            __m256 d2_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d2 + i * 2)));
            __m256 d3_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d3 + i * 2)));
            __m256 d4_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d4 + i * 2)));
            __m256 d5_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d5 + i * 2)));
            __m256 d6_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d6 + i * 2)));
            __m256 d7_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d7 + i * 2)));

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // Euclidean Step: acc = (Q - D)^2 + acc
            // Calculate difference (Q - D)
            __m256 diff0 = _mm256_sub_ps(q, d0_vec);
            __m256 diff1 = _mm256_sub_ps(q, d1_vec);
            __m256 diff2 = _mm256_sub_ps(q, d2_vec);
            __m256 diff3 = _mm256_sub_ps(q, d3_vec);
            __m256 diff4 = _mm256_sub_ps(q, d4_vec);
            __m256 diff5 = _mm256_sub_ps(q, d5_vec);
            __m256 diff6 = _mm256_sub_ps(q, d6_vec);
            __m256 diff7 = _mm256_sub_ps(q, d7_vec);

            // FMA: acc = (diff * diff) + acc  (accumulates the squared difference)
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
            acc4 = _mm256_fmadd_ps(diff4, diff4, acc4);
            acc5 = _mm256_fmadd_ps(diff5, diff5, acc5);
            acc6 = _mm256_fmadd_ps(diff6, diff6, acc6);
            acc7 = _mm256_fmadd_ps(diff7, diff7, acc7);
        }

        // --- 4. Horizontal Sum (Reduction) ---
        // The accumulators (acc0-acc7) hold 8 partial sums. They must be summed horizontally.

        // Helper function for horizontal reduction (sums all 8 elements in an __m256 register)
        auto horizontal_sum_avx2 = [](__m256 v) -> float {
            // Step 1: Horizontal add
            __m256 hsum = _mm256_hadd_ps(v, v);

            // Step 2: Second horizontal add
            hsum = _mm256_hadd_ps(hsum, hsum);

            // Step 3: Extract the 128-bit halves and add them (V0..V3 + V4..V7)
            __m128 hsum_128 = _mm256_extractf128_ps(hsum, 1); // Get upper 128 bits
            __m128 hsum_rest = _mm256_castps256_ps128(hsum); // Get lower 128 bits

            // Step 4: Add the two 128-bit sums together
            __m128 final_sum = _mm_add_ss(hsum_128, hsum_rest);

            // Step 5: Extract the result
            return _mm_cvtss_f32(final_sum);
        };

        float dist_sq0 = horizontal_sum_avx2(acc0);
        float dist_sq1 = horizontal_sum_avx2(acc1);
        float dist_sq2 = horizontal_sum_avx2(acc2);
        float dist_sq3 = horizontal_sum_avx2(acc3);
        float dist_sq4 = horizontal_sum_avx2(acc4);
        float dist_sq5 = horizontal_sum_avx2(acc5);
        float dist_sq6 = horizontal_sum_avx2(acc6);
        float dist_sq7 = horizontal_sum_avx2(acc7);

        // --- 5. Scalar Tail ---
        // Handle the remaining elements (dim % 8)
        for (; i < dim; i++) {
            // Load and convert
            const float qv = query[i];
            float tv0 = (float)*((const _Float16 *)(d0 + i * 2));
            float tv1 = (float)*((const _Float16 *)(d1 + i * 2));
            float tv2 = (float)*((const _Float16 *)(d2 + i * 2));
            float tv3 = (float)*((const _Float16 *)(d3 + i * 2));
            float tv4 = (float)*((const _Float16 *)(d4 + i * 2));
            float tv5 = (float)*((const _Float16 *)(d5 + i * 2));
            float tv6 = (float)*((const _Float16 *)(d6 + i * 2));
            float tv7 = (float)*((const _Float16 *)(d7 + i * 2));

            // Accumulate squared difference
            dist_sq0 += (qv - tv0) * (qv - tv0);
            dist_sq1 += (qv - tv1) * (qv - tv1);
            dist_sq2 += (qv - tv2) * (qv - tv2);
            dist_sq3 += (qv - tv3) * (qv - tv3);
            dist_sq4 += (qv - tv4) * (qv - tv4);
            dist_sq5 += (qv - tv5) * (qv - tv5);
            dist_sq6 += (qv - tv6) * (qv - tv6);
            dist_sq7 += (qv - tv7) * (qv - tv7);
        }

        // --- 6. Final Distance Calculation (Output is the raw Euclidean Distance) ---
        // Distance = sqrt(Distance^2)
        score0 = sqrtf(dist_sq0);
        score1 = sqrtf(dist_sq1);
        score2 = sqrtf(dist_sq2);
        score3 = sqrtf(dist_sq3);
        score4 = sqrtf(dist_sq4);
        score5 = sqrtf(dist_sq5);
        score6 = sqrtf(dist_sq6);
        score7 = sqrtf(dist_sq7);
    }

    static void batchEuclidian4FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0, const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2, const uint8_t* RESTRICT d3,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3) {

        // Accumulators (4 vectors, one for each target)
        // Initialize all 8 FP32 elements in the __m256 registers to 0.0f
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // The loop processes 8 elements (256 bits) at a time.
        const size_t dim_aligned = dim & ~7;
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            __m256 q = _mm256_loadu_ps(query + i);

            // --- 2. Load and Convert Target Vectors (4 targets * 8 FP16 values each) ---
            // Use F16C's _mm256_cvtph_ps to convert 8 FP16 values (128 bits)
            // into 8 FP32 values (256 bits) in one instruction.

            __m256 d0_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d0 + i * 2)));
            __m256 d1_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d1 + i * 2)));
            __m256 d2_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d2 + i * 2)));
            __m256 d3_vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d3 + i * 2)));

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // Euclidean Step: acc = (Q - D)^2 + acc
            // Calculate difference (Q - D)
            __m256 diff0 = _mm256_sub_ps(q, d0_vec);
            __m256 diff1 = _mm256_sub_ps(q, d1_vec);
            __m256 diff2 = _mm256_sub_ps(q, d2_vec);
            __m256 diff3 = _mm256_sub_ps(q, d3_vec);

            // FMA: acc = (diff * diff) + acc  (accumulates the squared difference)
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        // --- 4. Horizontal Sum (Reduction) ---
        // The accumulators (acc0-acc3) hold 8 partial sums. They must be summed horizontally.

        // Helper function for horizontal reduction (sums all 8 elements in an __m256 register)
        auto horizontal_sum_avx2 = [](__m256 v) -> float {
            // Step 1: Horizontal add
            __m256 hsum = _mm256_hadd_ps(v, v);

            // Step 2: Second horizontal add
            hsum = _mm256_hadd_ps(hsum, hsum);

            // Step 3: Extract the 128-bit halves and add them (V0..V3 + V4..V7)
            __m128 hsum_128 = _mm256_extractf128_ps(hsum, 1); // Get upper 128 bits
            __m128 hsum_rest = _mm256_castps256_ps128(hsum); // Get lower 128 bits

            // Step 4: Add the two 128-bit sums together
            __m128 final_sum = _mm_add_ss(hsum_128, hsum_rest);

            // Step 5: Extract the result
            return _mm_cvtss_f32(final_sum);
        };

        float dist_sq0 = horizontal_sum_avx2(acc0);
        float dist_sq1 = horizontal_sum_avx2(acc1);
        float dist_sq2 = horizontal_sum_avx2(acc2);
        float dist_sq3 = horizontal_sum_avx2(acc3);

        // --- 5. Scalar Tail ---
        // Handle the remaining elements (dim % 8)
        for (; i < dim; i++) {
            // Load and convert
            const float qv = query[i];
            float tv0 = (float)*((const _Float16 *)(d0 + i * 2));
            float tv1 = (float)*((const _Float16 *)(d1 + i * 2));
            float tv2 = (float)*((const _Float16 *)(d2 + i * 2));
            float tv3 = (float)*((const _Float16 *)(d3 + i * 2));

            // Accumulate squared difference
            dist_sq0 += (qv - tv0) * (qv - tv0);
            dist_sq1 += (qv - tv1) * (qv - tv1);
            dist_sq2 += (qv - tv2) * (qv - tv2);
            dist_sq3 += (qv - tv3) * (qv - tv3);
        }

        // --- 6. Final Distance Calculation (Output is the raw Euclidean Distance) ---
        // Distance = sqrt(Distance^2)
        score0 = sqrtf(dist_sq0);
        score1 = sqrtf(dist_sq1);
        score2 = sqrtf(dist_sq2);
        score3 = sqrtf(dist_sq3);
    }
};

FP16L2SimilarityFunction FP16_L2_SIMIL_FUNC;

SimilarityFunction* SimilarityFunction::selectSimilarityFunction(NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
