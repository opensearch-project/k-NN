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

// Plain AVX512 build. BF16 inner product falls back to FP32 conversion + FMA,
// since native AVX512-BF16 instructions are not guaranteed on this target.
#include "avx512_common_simd_similarity_function.cpp"

//
//  BF16 — Follows the same pattern as FP16 above.
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock      = 8;
        constexpr int32_t elemPerLoad   = 16;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim   = dim - simdDim;

        // Precompute tail mask
        const __mmask16 tailMask = tailDim > 0 ? (__mmask16)((1U << tailDim) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + (i + elemPerLoad), 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            if (tailDim > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                __m512 v = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

#include "avx512_common_registration.cpp"
