#include <algorithm>
#include <immintrin.h>
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
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        // The loop processes 16 elements (512 bits) at a time.
        const size_t dim_aligned = dim & ~15; // Align to 16
        size_t i = 0;

        for (; i < dim_aligned; i += 16) {
            // --- 1. Load Query (16 FP32 values) ---
            // Aligned load, leveraging the 64-byte alignment guarantee.
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            __m512 q = _mm512_load_ps(aligned_q_ptr);

            // --- 2. Load and Convert Target Vectors (8 targets * 16 FP16 values each) ---
            // _mm512_cvtph_ps loads 256 bits (16 FP16 values) and converts them to 512 bits (16 FP32 values).

            __m256i h0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(h0);
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(h1);
            __m256i h2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(h2);
            __m256i h3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(h3);
            __m256i h4 = _mm256_loadu_si256((const __m256i *)(d4 + i * 2));
            __m512 d4_vec = _mm512_cvtph_ps(h4);
            __m256i h5 = _mm256_loadu_si256((const __m256i *)(d5 + i * 2));
            __m512 d5_vec = _mm512_cvtph_ps(h5);
            __m256i h6 = _mm256_loadu_si256((const __m256i *)(d6 + i * 2));
            __m512 d6_vec = _mm512_cvtph_ps(h6);
            __m256i h7 = _mm256_loadu_si256((const __m256i *)(d7 + i * 2));
            __m512 d7_vec = _mm512_cvtph_ps(h7);

            // --- 3. Compute Product and Accumulate (Inner Product) ---
            // Inner Product Step: acc = (Q * D) + acc using FMA

            // accX = accX + (Q * dX_vec)
            acc0 = _mm512_fmadd_ps(q, d0_vec, acc0);
            acc1 = _mm512_fmadd_ps(q, d1_vec, acc1);
            acc2 = _mm512_fmadd_ps(q, d2_vec, acc2);
            acc3 = _mm512_fmadd_ps(q, d3_vec, acc3);
            acc4 = _mm512_fmadd_ps(q, d4_vec, acc4);
            acc5 = _mm512_fmadd_ps(q, d5_vec, acc5);
            acc6 = _mm512_fmadd_ps(q, d6_vec, acc6);
            acc7 = _mm512_fmadd_ps(q, d7_vec, acc7);
        }

        // --- 4. Masked Tail Handling (AVX-512) ---
        // If dim is not a multiple of 16, handle the remaining elements using masking.
        if (i < dim) {
            const int remaining_elements = dim - i;
            // Create a mask (e.g., if 3 remain: 0b111...0)
            __mmask16 tail_mask = _mm512_int2mask((1 << remaining_elements) - 1);

            // Load Query (masked load: loads only valid elements, rest are zeroed/untouched)
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            __m512 q = _mm512_mask_load_ps(_mm512_setzero_ps(), tail_mask, aligned_q_ptr);

            // Load and Convert Targets (masked operations)
            __m256i h0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(h0);
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(h1);
            __m256i h2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(h2);
            __m256i h3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(h3);
            __m256i h4 = _mm256_loadu_si256((const __m256i *)(d4 + i * 2));
            __m512 d4_vec = _mm512_cvtph_ps(h4);
            __m256i h5 = _mm256_loadu_si256((const __m256i *)(d5 + i * 2));
            __m512 d5_vec = _mm512_cvtph_ps(h5);
            __m256i h6 = _mm256_loadu_si256((const __m256i *)(d6 + i * 2));
            __m512 d6_vec = _mm512_cvtph_ps(h6);
            __m256i h7 = _mm256_loadu_si256((const __m256i *)(d7 + i * 2));
            __m512 d7_vec = _mm512_cvtph_ps(h7);

            // Compute Product and Masked Accumulation
            // _mm512_mask_fmadd_ps: acc = (mask) ? (acc + Q * D) : acc

            acc0 = _mm512_mask_fmadd_ps(acc0, tail_mask, q, d0_vec);
            acc1 = _mm512_mask_fmadd_ps(acc1, tail_mask, q, d1_vec);
            acc2 = _mm512_mask_fmadd_ps(acc2, tail_mask, q, d2_vec);
            acc3 = _mm512_mask_fmadd_ps(acc3, tail_mask, q, d3_vec);
            acc4 = _mm512_mask_fmadd_ps(acc4, tail_mask, q, d4_vec);
            acc5 = _mm512_mask_fmadd_ps(acc5, tail_mask, q, d5_vec);
            acc6 = _mm512_mask_fmadd_ps(acc6, tail_mask, q, d6_vec);
            acc7 = _mm512_mask_fmadd_ps(acc7, tail_mask, q, d7_vec);
        }

        // --- 5. Final Reduction (Horizontal Sum) ---
        // dist_sqX is the L2^2 sum
        score0 = _mm512_reduce_add_ps(acc0);
        score1 = _mm512_reduce_add_ps(acc1);
        score2 = _mm512_reduce_add_ps(acc2);
        score3 = _mm512_reduce_add_ps(acc3);
        score4 = _mm512_reduce_add_ps(acc4);
        score5 = _mm512_reduce_add_ps(acc5);
        score6 = _mm512_reduce_add_ps(acc6);
        score7 = _mm512_reduce_add_ps(acc7);
    }

    void batchInnerProduct4FP16Targets(
        const float * RESTRICT query,
        const uint8_t * RESTRICT d0,
        const uint8_t * RESTRICT d1,
        const uint8_t * RESTRICT d2,
        const uint8_t * RESTRICT d3,
        size_t dim,
        float& score0,
        float& score1,
        float& score2,
        float& score3) {

        // Accumulators (4 vectors, one for each target) for the Dot Product sum.
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        // The loop processes 16 elements (512 bits) at a time.
        const size_t dim_aligned = dim & ~15; // Align to 16
        size_t i = 0;

        for (; i < dim_aligned; i += 16) {
            // --- 1. Load Query (16 FP32 values) ---
            // Aligned load, leveraging the 64-byte alignment guarantee.
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            __m512 q = _mm512_load_ps(aligned_q_ptr);

            // --- 2. Load and Convert Target Vectors (4 targets * 16 FP16 values each) ---
            // _mm512_cvtph_ps loads 256 bits (16 FP16 values) and converts them to 512 bits (16 FP32 values).

            __m256i fp16_data_d0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(fp16_data_d0);

            __m256i fp16_data_d1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(fp16_data_d1);

            __m256i fp16_data_d2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(fp16_data_d2);

            __m256i fp16_data_d3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(fp16_data_d3);

            // --- 3. Compute Product and Accumulate (Inner Product) ---
            // Inner Product Step: acc = (Q * D) + acc using FMA

            // accX = accX + (Q * dX_vec)
            acc0 = _mm512_fmadd_ps(q, d0_vec, acc0);
            acc1 = _mm512_fmadd_ps(q, d1_vec, acc1);
            acc2 = _mm512_fmadd_ps(q, d2_vec, acc2);
            acc3 = _mm512_fmadd_ps(q, d3_vec, acc3);
        }

        // --- 4. Masked Tail Handling (AVX-512) ---
        // If dim is not a multiple of 16, handle the remaining elements using masking.
        if (i < dim) {
            const int remaining_elements = dim - i;
            // Create a mask (e.g., if 3 remain: 0b111...0)
            __mmask16 tail_mask = _mm512_int2mask((1 << remaining_elements) - 1);

            // Load Query (masked load: loads only valid elements, rest are zeroed/untouched)
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            __m512 q = _mm512_mask_load_ps(_mm512_setzero_ps(), tail_mask, aligned_q_ptr);

            // Load and Convert Targets (masked operations)
            __m256i h0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(h0);
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(h1);
            __m256i h2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(h2);
            __m256i h3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(h3);

            // Compute Product and Masked Accumulation
            // _mm512_mask_fmadd_ps: acc = (mask) ? (acc + Q * D) : acc

            acc0 = _mm512_mask_fmadd_ps(acc0, tail_mask, q, d0_vec);
            acc1 = _mm512_mask_fmadd_ps(acc1, tail_mask, q, d1_vec);
            acc2 = _mm512_mask_fmadd_ps(acc2, tail_mask, q, d2_vec);
            acc3 = _mm512_mask_fmadd_ps(acc3, tail_mask, q, d3_vec);
        }

        // --- 5. Final Reduction (Horizontal Sum) ---
        score0 = _mm512_reduce_add_ps(acc0);
        score1 = _mm512_reduce_add_ps(acc1);
        score2 = _mm512_reduce_add_ps(acc2);
        score3 = _mm512_reduce_add_ps(acc3);
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

        // Query is 64 bytes aligned. Giving compiler a hint for good.
        BUILTIN_ASSUME_ALIGNED(query, 64);

        // Accumulators (4 vectors, one for each target) for L2 Squared
        // Initialize all 16 FP32 elements in the __m512 registers to 0.0f
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        // The loop processes 16 elements (512 bits) at a time.
        const size_t dim_aligned = dim & ~15; // Align to 16
        size_t i = 0;

        for (; i < dim_aligned; i += 16) {
            // --- 1. Load Query (16 FP32 values) ---
            __m512 q = _mm512_load_ps(query + i);

            // --- 2. Load and Convert Target Vectors (4 targets * 16 FP16 values each) ---
            // Use F16C's _mm512_cvtph_ps to convert 16 FP16 values (256 bits)
            // into 16 FP32 values (512 bits) in one instruction.
            __m256i fp16_data_d0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0 = _mm512_cvtph_ps(fp16_data_d0);
            __m256i fp16_data_d1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1 = _mm512_cvtph_ps(fp16_data_d1);
            __m256i fp16_data_d2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2 = _mm512_cvtph_ps(fp16_data_d2);
            __m256i fp16_data_d3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3 = _mm512_cvtph_ps(fp16_data_d3);

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // Euclidean Step: acc = (Q - D)^2 + acc

            // Target 0
            __m512 diff0 = _mm512_sub_ps(q, d0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            // Target 1
            __m512 diff1 = _mm512_sub_ps(q, d1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            // Target 2
            __m512 diff2 = _mm512_sub_ps(q, d2);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            // Target 3
            __m512 diff3 = _mm512_sub_ps(q, d3);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        // --- 4. Final Reduction (Horizontal Sum for L2^2) ---
        // Use the optimized AVX-512 intrinsic to sum all 16 values in the register,
        // yielding the scalar squared distance.

        float dist_sq0 = _mm512_reduce_add_ps(acc0);
        float dist_sq1 = _mm512_reduce_add_ps(acc1);
        float dist_sq2 = _mm512_reduce_add_ps(acc2);
        float dist_sq3 = _mm512_reduce_add_ps(acc3);

        // --- 5. Scalar Tail (L2^2) ---
        // Handle the remaining elements (dim % 16)
        for (; i < dim; i++) {
            const float qv = query[i];

            // Load and convert target vector elements
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

    static void batchEuclidian4FP16Targets(
        const float* RESTRICT query,
        const uint8_t* RESTRICT d0, const uint8_t* RESTRICT d1,
        const uint8_t* RESTRICT d2, const uint8_t* RESTRICT d3,
        size_t dim,
        float& score0, float& score1, float& score2, float& score3) {

        // Accumulators (4 vectors, one for each target) for L2 squared (Distance^2)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        // The loop processes 16 elements (512 bits) at a time.
        const size_t dim_aligned = dim & ~15; // Align to 16
        size_t i = 0;

        for (; i < dim_aligned; i += 16) {
            // --- 1. Load Query (16 FP32 values) ---
            __m512 q = _mm512_loadu_ps(query + i);

            // --- 2. Load and Convert Target Vectors (4 targets * 16 FP16 values each) ---
            __m256i fp16_data_d0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(fp16_data_d0);
            __m256i fp16_data_d1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(fp16_data_d1);
            __m256i fp16_data_d2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(fp16_data_d2);
            __m256i fp16_data_d3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(fp16_data_d3);

            // --- 3. Compute Squared Difference and Accumulate (L2^2) ---
            // acc = acc + (Q - D)^2 using FMA: acc = acc + diff * diff

            __m512 diff0 = _mm512_sub_ps(q, d0_vec);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);

            __m512 diff1 = _mm512_sub_ps(q, d1_vec);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);

            __m512 diff2 = _mm512_sub_ps(q, d2_vec);
            acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);

            __m512 diff3 = _mm512_sub_ps(q, d3_vec);
            acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
        }

        // --- 4. Masked Tail Handling (AVX-512) ---
        // If dim is not a multiple of 16, handle the remaining elements using masking.
        if (i < dim) {
            const int remaining_elements = dim - i;
            // Create a mask (e.g., if 3 remain: 0b111...0)
            __mmask16 tail_mask = _mm512_int2mask((1 << remaining_elements) - 1);

            // Load Query (masked load: loads only valid elements, rest are zeroed/untouched)
            const float *aligned_q_ptr = BUILTIN_ASSUME_ALIGNED(query + i, 64);
            // Use mask_load_ps to load only the required elements from the query
            __m512 q = _mm512_mask_load_ps(_mm512_setzero_ps(), tail_mask, aligned_q_ptr);

            // Load and Convert Targets (masked operations)
            // Load the FP16 data (unaligned load is fine as long as we mask the final result).
            __m256i h0 = _mm256_loadu_si256((const __m256i *)(d0 + i * 2));
            __m512 d0_vec = _mm512_cvtph_ps(h0);
            __m256i h1 = _mm256_loadu_si256((const __m256i *)(d1 + i * 2));
            __m512 d1_vec = _mm512_cvtph_ps(h1);
            __m256i h2 = _mm256_loadu_si256((const __m256i *)(d2 + i * 2));
            __m512 d2_vec = _mm512_cvtph_ps(h2);
            __m256i h3 = _mm256_loadu_si256((const __m256i *)(d3 + i * 2));
            __m512 d3_vec = _mm512_cvtph_ps(h3);

            // Compute Squared Difference and Masked Accumulation
            // _mm512_mask_fmadd_ps: Result[i] = (mask[i]) ? (acc[i] + diff[i] * diff[i]) : acc[i]

            __m512 diff0 = _mm512_sub_ps(q, d0_vec);
            acc0 = _mm512_mask_fmadd_ps(acc0, tail_mask, diff0, diff0);

            __m512 diff1 = _mm512_sub_ps(q, d1_vec);
            acc1 = _mm512_mask_fmadd_ps(acc1, tail_mask, diff1, diff1);

            __m512 diff2 = _mm512_sub_ps(q, d2_vec);
            acc2 = _mm512_mask_fmadd_ps(acc2, tail_mask, diff2, diff2);

            __m512 diff3 = _mm512_sub_ps(q, d3_vec);
            acc3 = _mm512_mask_fmadd_ps(acc3, tail_mask, diff3, diff3);
        }

        // --- 5. Final Reduction (Horizontal Sum) and Square Root ---
        // dist_sqX is the L2^2 sum
        float dist_sq0 = _mm512_reduce_add_ps(acc0);
        float dist_sq1 = _mm512_reduce_add_ps(acc1);
        float dist_sq2 = _mm512_reduce_add_ps(acc2);
        float dist_sq3 = _mm512_reduce_add_ps(acc3);

        // --- 6. Final L2 Distance (Square Root) ---
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
