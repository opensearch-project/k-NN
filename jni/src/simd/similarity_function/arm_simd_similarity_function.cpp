#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdint.h>

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

        // Accumulators (8 vectors, one for each target) for the Dot Product sum.
        // Each accumulator holds 4 partial sums.
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        float32x4_t acc5 = vdupq_n_f32(0.0f);
        float32x4_t acc6 = vdupq_n_f32(0.0f);
        float32x4_t acc7 = vdupq_n_f32(0.0f);

        // The loop processes 8 elements (two 128-bit NEON registers) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Use the macro for alignment hint, the intrinsic is vld1q_f32.
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            float32x4_t q0 = vld1q_f32(aligned_q_ptr);      // First 4 elements
            float32x4_t q1 = vld1q_f32(aligned_q_ptr + 4);  // Next 4 elements

            // --- 2. Load and Convert Target Vectors (8 targets * 8 FP16 values each) ---

            // Load 8 FP16 values (16 bytes) and convert to two FP32x4 vectors (32 bytes total)
            #define LOAD_CONVERT_TARGET(D_IN, D_LO, D_HI) \
                float16x8_t D_IN = vld1q_f16((const __fp16 *)(D_IN + i * 2)); \
                float32x4_t D_LO = vcvt_f32_f16(vget_low_f16(D_IN)); \
                float32x4_t D_HI = vcvt_f32_f16(vget_high_f16(D_IN));

            // Note: Using explicit variable names instead of macros for readability.

            // Target 0
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));

            // Target 1
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));

            // Target 2
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
            float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));

            // Target 3
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
            float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
            float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

            // Target 4
            float16x8_t h4 = vld1q_f16((const __fp16 *)(d4 + i * 2));
            float32x4_t d4_lo = vcvt_f32_f16(vget_low_f16(h4));
            float32x4_t d4_hi = vcvt_f32_f16(vget_high_f16(h4));

            // Target 5
            float16x8_t h5 = vld1q_f16((const __fp16 *)(d5 + i * 2));
            float32x4_t d5_lo = vcvt_f32_f16(vget_low_f16(h5));
            float32x4_t d5_hi = vcvt_f32_f16(vget_high_f16(h5));

            // Target 6
            float16x8_t h6 = vld1q_f16((const __fp16 *)(d6 + i * 2));
            float32x4_t d6_lo = vcvt_f32_f16(vget_low_f16(h6));
            float32x4_t d6_hi = vcvt_f32_f16(vget_high_f16(h6));

            // Target 7
            float16x8_t h7 = vld1q_f16((const __fp16 *)(d7 + i * 2));
            float32x4_t d7_lo = vcvt_f32_f16(vget_low_f16(h7));
            float32x4_t d7_hi = vcvt_f32_f16(vget_high_f16(h7));

            // Post-load prefetch: next 8 elements
            // By the time in the next loop,
            if ((i + 8) < dim) {
                __builtin_prefetch(query + i + 8);
                __builtin_prefetch(d0 + (i + 8) * 2);
                __builtin_prefetch(d1 + (i + 8) * 2);
                __builtin_prefetch(d2 + (i + 8) * 2);
                __builtin_prefetch(d3 + (i + 8) * 2);
                __builtin_prefetch(d4 + (i + 8) * 2);
                __builtin_prefetch(d5 + (i + 8) * 2);
                __builtin_prefetch(d6 + (i + 8) * 2);
                __builtin_prefetch(d7 + (i + 8) * 2);
            }

            // --- 3. Compute Product and Accumulate (Inner Product) ---
            // Inner Product Step: acc = (Q * D) + acc using FMA (vfmaq_f32)

            // accX = accX + (q0 * dX_lo) + (q1 * dX_hi)
            acc0 = vfmaq_f32(acc0, q0, d0_lo);
            acc0 = vfmaq_f32(acc0, q1, d0_hi);

            acc1 = vfmaq_f32(acc1, q0, d1_lo);
            acc1 = vfmaq_f32(acc1, q1, d1_hi);

            acc2 = vfmaq_f32(acc2, q0, d2_lo);
            acc2 = vfmaq_f32(acc2, q1, d2_hi);

            acc3 = vfmaq_f32(acc3, q0, d3_lo);
            acc3 = vfmaq_f32(acc3, q1, d3_hi);

            acc4 = vfmaq_f32(acc4, q0, d4_lo);
            acc4 = vfmaq_f32(acc4, q1, d4_hi);

            acc5 = vfmaq_f32(acc5, q0, d5_lo);
            acc5 = vfmaq_f32(acc5, q1, d5_hi);

            acc6 = vfmaq_f32(acc6, q0, d6_lo);
            acc6 = vfmaq_f32(acc6, q1, d6_hi);

            acc7 = vfmaq_f32(acc7, q0, d7_lo);
            acc7 = vfmaq_f32(acc7, q1, d7_hi);
        }

        // --- 4. Scalar Tail Handling ---
        // Handle the remaining elements (dim % 8) using scalar operations.
        float final_sum0 = vaddvq_f32(acc0);
        float final_sum1 = vaddvq_f32(acc1);
        float final_sum2 = vaddvq_f32(acc2);
        float final_sum3 = vaddvq_f32(acc3);
        float final_sum4 = vaddvq_f32(acc4);
        float final_sum5 = vaddvq_f32(acc5);
        float final_sum6 = vaddvq_f32(acc6);
        float final_sum7 = vaddvq_f32(acc7);

        for (; i < dim; i++) {
            const float qv = query[i];

            // Load FP16 targets and convert to float (uses the explicit cast operator)
            float tv0 = (float)*((const __fp16 *)(d0 + i * 2));
            float tv1 = (float)*((const __fp16 *)(d1 + i * 2));
            float tv2 = (float)*((const __fp16 *)(d2 + i * 2));
            float tv3 = (float)*((const __fp16 *)(d3 + i * 2));
            float tv4 = (float)*((const __fp16 *)(d4 + i * 2));
            float tv5 = (float)*((const __fp16 *)(d5 + i * 2));
            float tv6 = (float)*((const __fp16 *)(d6 + i * 2));
            float tv7 = (float)*((const __fp16 *)(d7 + i * 2));

            final_sum0 += qv * tv0;
            final_sum1 += qv * tv1;
            final_sum2 += qv * tv2;
            final_sum3 += qv * tv3;
            final_sum4 += qv * tv4;
            final_sum5 += qv * tv5;
            final_sum6 += qv * tv6;
            final_sum7 += qv * tv7;
        }

        // --- 5. Final Score ---
        score0 = final_sum0;
        score1 = final_sum1;
        score2 = final_sum2;
        score3 = final_sum3;
        score4 = final_sum4;
        score5 = final_sum5;
        score6 = final_sum6;
        score7 = final_sum7;
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

        // Accumulators (4 vectors, one for each target) for the Dot Product sum.
        // Each accumulator holds 4 partial sums.
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        // The loop processes 8 elements (two 128-bit NEON registers) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Use the macro for alignment hint, the intrinsic is vld1q_f32.
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            float32x4_t q0 = vld1q_f32(aligned_q_ptr);      // First 4 elements
            float32x4_t q1 = vld1q_f32(aligned_q_ptr + 4);  // Next 4 elements

            // --- 2. Load and Convert Target Vectors (4 targets * 8 FP16 values each) ---

            // Target 0
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));

            // Target 1
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));

            // Target 2
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
            float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));

            // Target 3
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
            float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
            float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

            // Post-load prefetch: next 8 elements
            // By the time in the next loop,
            if ((i + 8) < dim) {
                __builtin_prefetch(query + i + 8);
                __builtin_prefetch(d0 + (i + 8) * 2);
                __builtin_prefetch(d1 + (i + 8) * 2);
                __builtin_prefetch(d2 + (i + 8) * 2);
                __builtin_prefetch(d3 + (i + 8) * 2);
            }

            // --- 3. Compute Product and Accumulate (Inner Product) ---
            // Inner Product Step: acc = (Q * D) + acc using FMA (vfmaq_f32)

            // accX = accX + (q0 * dX_lo) + (q1 * dX_hi)
            acc0 = vfmaq_f32(acc0, q0, d0_lo);
            acc0 = vfmaq_f32(acc0, q1, d0_hi);

            acc1 = vfmaq_f32(acc1, q0, d1_lo);
            acc1 = vfmaq_f32(acc1, q1, d1_hi);

            acc2 = vfmaq_f32(acc2, q0, d2_lo);
            acc2 = vfmaq_f32(acc2, q1, d2_hi);

            acc3 = vfmaq_f32(acc3, q0, d3_lo);
            acc3 = vfmaq_f32(acc3, q1, d3_hi);
        }

        // --- 4. Scalar Tail Handling ---
        // Handle the remaining elements (dim % 8) using scalar operations.
        float final_sum0 = vaddvq_f32(acc0);
        float final_sum1 = vaddvq_f32(acc1);
        float final_sum2 = vaddvq_f32(acc2);
        float final_sum3 = vaddvq_f32(acc3);

        for (; i < dim; i++) {
            const float qv = query[i];

            // Load FP16 targets and convert to float (uses the explicit cast operator)
            float tv0 = (float)*((const __fp16 *)(d0 + i * 2));
            float tv1 = (float)*((const __fp16 *)(d1 + i * 2));
            float tv2 = (float)*((const __fp16 *)(d2 + i * 2));
            float tv3 = (float)*((const __fp16 *)(d3 + i * 2));

            final_sum0 += qv * tv0;
            final_sum1 += qv * tv1;
            final_sum2 += qv * tv2;
            final_sum3 += qv * tv3;
        }

        // --- 5. Final Score ---
        score0 = final_sum0;
        score1 = final_sum1;
        score2 = final_sum2;
        score3 = final_sum3;
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

        // Accumulators (8 vectors, one for each target) for the L2 Squared sum.
        // Each accumulator holds 4 partial sums.
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        float32x4_t acc5 = vdupq_n_f32(0.0f);
        float32x4_t acc6 = vdupq_n_f32(0.0f);
        float32x4_t acc7 = vdupq_n_f32(0.0f);

        // The loop processes 8 elements (two 128-bit NEON registers) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Use the macro for alignment hint. q0 = elements [i..i+3], q1 = elements [i+4..i+7].
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            float32x4_t q0 = vld1q_f32(aligned_q_ptr);
            float32x4_t q1 = vld1q_f32(aligned_q_ptr + 4);

            // --- 2. Load and Convert Target Vectors (8 targets * 8 FP16 values each) ---

            // Target 0
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));

            // Target 1
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));

            // Target 2
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
            float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));

            // Target 3
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
            float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
            float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

            // Target 4
            float16x8_t h4 = vld1q_f16((const __fp16 *)(d4 + i * 2));
            float32x4_t d4_lo = vcvt_f32_f16(vget_low_f16(h4));
            float32x4_t d4_hi = vcvt_f32_f16(vget_high_f16(h4));

            // Target 5
            float16x8_t h5 = vld1q_f16((const __fp16 *)(d5 + i * 2));
            float32x4_t d5_lo = vcvt_f32_f16(vget_low_f16(h5));
            float32x4_t d5_hi = vcvt_f32_f16(vget_high_f16(h5));

            // Target 6
            float16x8_t h6 = vld1q_f16((const __fp16 *)(d6 + i * 2));
            float32x4_t d6_lo = vcvt_f32_f16(vget_low_f16(h6));
            float32x4_t d6_hi = vcvt_f32_f16(vget_high_f16(h6));

            // Target 7
            float16x8_t h7 = vld1q_f16((const __fp16 *)(d7 + i * 2));
            float32x4_t d7_lo = vcvt_f32_f16(vget_low_f16(h7));
            float32x4_t d7_hi = vcvt_f32_f16(vget_high_f16(h7));

            // Post-load prefetch: next 8 elements
            // By the time in the next loop,
            if ((i + 8) < dim) {
                __builtin_prefetch(query + i + 8);
                __builtin_prefetch(d0 + (i + 8) * 2);
                __builtin_prefetch(d1 + (i + 8) * 2);
                __builtin_prefetch(d2 + (i + 8) * 2);
                __builtin_prefetch(d3 + (i + 8) * 2);
                __builtin_prefetch(d4 + (i + 8) * 2);
                __builtin_prefetch(d5 + (i + 8) * 2);
                __builtin_prefetch(d6 + (i + 8) * 2);
                __builtin_prefetch(d7 + (i + 8) * 2);
            }

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // L2^2 Step: acc = acc + (Q - D)^2 using VSUB and VFMA(acc, diff, diff)

            #define L2_ACCUMULATE(ACC, Q0, Q1, D_LO, D_HI) \
                {\
                float32x4_t diff_lo = vsubq_f32(Q0, D_LO); \
                ACC = vfmaq_f32(ACC, diff_lo, diff_lo); \
                float32x4_t diff_hi = vsubq_f32(Q1, D_HI); \
                ACC = vfmaq_f32(ACC, diff_hi, diff_hi); \
                }

            L2_ACCUMULATE(acc0, q0, q1, d0_lo, d0_hi)
            L2_ACCUMULATE(acc1, q0, q1, d1_lo, d1_hi)
            L2_ACCUMULATE(acc2, q0, q1, d2_lo, d2_hi)
            L2_ACCUMULATE(acc3, q0, q1, d3_lo, d3_hi)
            L2_ACCUMULATE(acc4, q0, q1, d4_lo, d4_hi)
            L2_ACCUMULATE(acc5, q0, q1, d5_lo, d5_hi)
            L2_ACCUMULATE(acc6, q0, q1, d6_lo, d6_hi)
            L2_ACCUMULATE(acc7, q0, q1, d7_lo, d7_hi)

            #undef L2_ACCUMULATE
        }

        // --- 4. Scalar Tail Handling ---
        // Handle the remaining elements (dim % 8) using scalar operations.
        float final_sum0 = vaddvq_f32(acc0);
        float final_sum1 = vaddvq_f32(acc1);
        float final_sum2 = vaddvq_f32(acc2);
        float final_sum3 = vaddvq_f32(acc3);
        float final_sum4 = vaddvq_f32(acc4);
        float final_sum5 = vaddvq_f32(acc5);
        float final_sum6 = vaddvq_f32(acc6);
        float final_sum7 = vaddvq_f32(acc7);

        for (; i < dim; i++) {
            const float qv = query[i];

            // Load FP16 targets and convert to float (uses the explicit cast operator)
            float tv0 = (float)*((const __fp16 *)(d0 + i * 2));
            float tv1 = (float)*((const __fp16 *)(d1 + i * 2));
            float tv2 = (float)*((const __fp16 *)(d2 + i * 2));
            float tv3 = (float)*((const __fp16 *)(d3 + i * 2));
            float tv4 = (float)*((const __fp16 *)(d4 + i * 2));
            float tv5 = (float)*((const __fp16 *)(d5 + i * 2));
            float tv6 = (float)*((const __fp16 *)(d6 + i * 2));
            float tv7 = (float)*((const __fp16 *)(d7 + i * 2));

            // Compute squared difference (Q - D)^2
            float diff0 = qv - tv0;
            float diff1 = qv - tv1;
            float diff2 = qv - tv2;
            float diff3 = qv - tv3;
            float diff4 = qv - tv4;
            float diff5 = qv - tv5;
            float diff6 = qv - tv6;
            float diff7 = qv - tv7;

            final_sum0 += diff0 * diff0;
            final_sum1 += diff1 * diff1;
            final_sum2 += diff2 * diff2;
            final_sum3 += diff3 * diff3;
            final_sum4 += diff4 * diff4;
            final_sum5 += diff5 * diff5;
            final_sum6 += diff6 * diff6;
            final_sum7 += diff7 * diff7;
        }

        // --- 5. Final Score (L2 Distance = sqrt(L2 Squared)) ---
        score0 = sqrtf(final_sum0);
        score1 = sqrtf(final_sum1);
        score2 = sqrtf(final_sum2);
        score3 = sqrtf(final_sum3);
        score4 = sqrtf(final_sum4);
        score5 = sqrtf(final_sum5);
        score6 = sqrtf(final_sum6);
        score7 = sqrtf(final_sum7);
    }

    static void batchEuclidian4FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0, const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2, const uint8_t* RESTRICT d3,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3) {

        // Accumulators (4 vectors, one for each target) for the L2 Squared sum.
        // Each accumulator holds 4 partial sums.
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        // The loop processes 8 elements (two 128-bit NEON registers) at a time.
        const size_t dim_aligned = dim & ~7; // Align to 8
        size_t i = 0;

        for (; i < dim_aligned; i += 8) {
            // --- 1. Load Query (8 FP32 values) ---
            // Use the macro for alignment hint. q0 = elements [i..i+3], q1 = elements [i+4..i+7].
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            float32x4_t q0 = vld1q_f32(aligned_q_ptr);
            float32x4_t q1 = vld1q_f32(aligned_q_ptr + 4);

            // --- 2. Load and Convert Target Vectors (4 targets * 8 FP16 values each) ---

            // Target 0
            float16x8_t h0 = vld1q_f16((const __fp16 *)(d0 + i * 2));
            float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));

            // Target 1
            float16x8_t h1 = vld1q_f16((const __fp16 *)(d1 + i * 2));
            float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));

            // Target 2
            float16x8_t h2 = vld1q_f16((const __fp16 *)(d2 + i * 2));
            float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
            float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));

            // Target 3
            float16x8_t h3 = vld1q_f16((const __fp16 *)(d3 + i * 2));
            float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
            float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

            // Post-load prefetch: next 8 elements
            // By the time in the next loop,
            if ((i + 8) < dim) {
                __builtin_prefetch(query + i + 8);
                __builtin_prefetch(d0 + (i + 8) * 2);
                __builtin_prefetch(d1 + (i + 8) * 2);
                __builtin_prefetch(d2 + (i + 8) * 2);
                __builtin_prefetch(d3 + (i + 8) * 2);
            }

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // L2^2 Step: acc = acc + (Q - D)^2 using VSUB and VFMA(acc, diff, diff)

            #define L2_ACCUMULATE(ACC, Q0, Q1, D_LO, D_HI) \
                {\
                float32x4_t diff_lo = vsubq_f32(Q0, D_LO); \
                ACC = vfmaq_f32(ACC, diff_lo, diff_lo); \
                float32x4_t diff_hi = vsubq_f32(Q1, D_HI); \
                ACC = vfmaq_f32(ACC, diff_hi, diff_hi); \
                }

            L2_ACCUMULATE(acc0, q0, q1, d0_lo, d0_hi)
            L2_ACCUMULATE(acc1, q0, q1, d1_lo, d1_hi)
            L2_ACCUMULATE(acc2, q0, q1, d2_lo, d2_hi)
            L2_ACCUMULATE(acc3, q0, q1, d3_lo, d3_hi)

            #undef L2_ACCUMULATE
        }

        // --- 4. Scalar Tail Handling ---
        // Handle the remaining elements (dim % 8) using scalar operations.
        float final_sum0 = vaddvq_f32(acc0);
        float final_sum1 = vaddvq_f32(acc1);
        float final_sum2 = vaddvq_f32(acc2);
        float final_sum3 = vaddvq_f32(acc3);

        for (; i < dim; i++) {
            const float qv = query[i];

            // Load FP16 targets and convert to float (uses the explicit cast operator)
            float tv0 = (float)*((const __fp16 *)(d0 + i * 2));
            float tv1 = (float)*((const __fp16 *)(d1 + i * 2));
            float tv2 = (float)*((const __fp16 *)(d2 + i * 2));
            float tv3 = (float)*((const __fp16 *)(d3 + i * 2));

            // Compute squared difference (Q - D)^2
            float diff0 = qv - tv0;
            float diff1 = qv - tv1;
            float diff2 = qv - tv2;
            float diff3 = qv - tv3;

            final_sum0 += diff0 * diff0;
            final_sum1 += diff1 * diff1;
            final_sum2 += diff2 * diff2;
            final_sum3 += diff3 * diff3;
        }

        // --- 5. Final Score (L2 Distance = sqrt(L2 Squared)) ---
        score0 = sqrtf(final_sum0);
        score1 = sqrtf(final_sum1);
        score2 = sqrtf(final_sum2);
        score3 = sqrtf(final_sum3);
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
