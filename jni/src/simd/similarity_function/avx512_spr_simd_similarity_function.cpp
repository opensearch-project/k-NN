#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"



//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        // How many vectors are processed so far?
        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        // Use 8 to keep the register pressure low
        constexpr int32_t vecBlock = 8;
        // Maximum number of elements to load at the same time
        constexpr int32_t elemPerLoad = 16;
        // Tracking accumulated summation per each vector
        // FYI : IP = Sum(v1[i] * v2[i])
        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            // Initialize sum variables
            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                const int32_t rem = dim - i;
                __mmask16 mask = rem < elemPerLoad ? (__mmask16)((1U << rem) - 1) : 0xFFFF;

                // Load query vector
                __m512 q0 = _mm512_maskz_loadu_ps(mask, queryPtr + i);

                __m512 vRegs[vecBlock];
                // Convert N FP16 values to FP32 values per each vector.
                // vRegs[i] will hold N FP32 converted values from ith vector.
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, vectors[v] + 2 * i));
                }

                // Trigger prefetch for the next elements (For the next iteration: +16 elements = +32 bytes)
                // While we're doing FMA operation, this will help it pull the next elements to fit into L1 cache.
                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                // FMA Operation e.g. IP = IP + q[i] * v[i]
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
                }
            }

            // __m512 have 16 FP32 values.
            // __m512_reduce_add_ps is summing the values stored in __m512.
            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            // Get vector
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                const int32_t rem = dim - i;
                __mmask16 mask = rem < elemPerLoad ? (__mmask16)((1U << rem) - 1) : 0xFFFF;

                // Have N FP32 values from query
                __m512 q = _mm512_maskz_loadu_ps(mask, queryPtr + i);
                // Have N FP32 values from vector
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, vecPtr + 2 * i));
                // Do FMA e.g. IP = IP + q[i] * v[i]
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            // __m512 have 16 FP32 values.
            // __m512_reduce_add_ps is summing the values stored in __m512.
            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        // Now, convert score values to Max-IP score scheme that Lucene uses.
        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 16;
        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                const int32_t rem = dim - i;
                __mmask16 mask = rem < elemPerLoad ? (__mmask16)((1U << rem) - 1) : 0xFFFF;

                // LOAD & CONVERT
                __m512 q0 = _mm512_maskz_loadu_ps(mask, queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, vectors[v] + 2 * i));
                }

                // TRIGGER PREFETCH
                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                // L2 MATH: (q - v)^2 + sum
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    // Compute difference: diff = q - v
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    // Square and Accumulate: sum = (diff * diff) + sum
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const _Float16* vecPtr = (const _Float16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                const int32_t rem = dim - i;
                __mmask16 mask = rem < elemPerLoad ? (__mmask16)((1U << rem) - 1) : 0xFFFF;

                __m512 q = _mm512_maskz_loadu_ps(mask, queryPtr + i);
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, vecPtr + i));

                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }
           scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
       }

       BulkScoreTransformFunc(scores, numVectors);
   }
};



//
// FP16
//
// 1. Max IP
AVX512SPRFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512SPRFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
