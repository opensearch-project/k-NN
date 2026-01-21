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
                                   const int32_t numVectors) final {

        int32_t processedCount = 0;
        const _Float16* vectors[16];
        const _Float16* queryPtr =
            reinterpret_cast<const _Float16*>(srchContext->queryVectorSimdAligned);

        const int32_t dim = srchContext->dimension;
        const int32_t mainDim = dim & ~63;  // multiple of 64

        // Prefetch query once for the whole call
        __builtin_prefetch(queryPtr, 0, 3);

        // MAIN LOOP: process 16 vectors at once
        for (; processedCount <= numVectors - 16; processedCount += 16) {
            srchContext->getVectorPointersInBulk(
                reinterpret_cast<uint8_t**>(vectors),
                &internalVectorIds[processedCount],
                16);

            // 16 FP32 accumulators
            __m512 sum[16];
    #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sum[i] = _mm512_setzero_ps();
            }

            // Dimension loop, unrolled by 64 FP16
            for (int32_t i = 0; i < mainDim; i += 64) {
                __m512h q0 = _mm512_load_ph(queryPtr + i);
                __m512h q1 = _mm512_load_ph(queryPtr + i + 32);

    #pragma unroll
                for (int32_t v = 0; v < 16; ++v) {
                    const _Float16* vec = vectors[v];

                    __m512h v0 = _mm512_loadu_ph(vec + i);
                    __m512h v1 = _mm512_loadu_ph(vec + i + 32);

                    sum[v] = _mm512_fmadd_ph(q0, v0, sum[v]);
                    sum[v] = _mm512_fmadd_ph(q1, v1, sum[v]);
                }
            }

            // Tail for dimension (single masked iteration at most)
            if (mainDim < dim) {
                const int32_t rem = dim - mainDim;
                const __mmask32 mask = (1u << rem) - 1;

                __m512h q = _mm512_maskz_load_ph(mask, queryPtr + mainDim);

    #pragma unroll
                for (int32_t v = 0; v < 16; ++v) {
                    __m512h vec = _mm512_maskz_loadu_ph(mask, vectors[v] + mainDim);
                    sum[v] = _mm512_fmadd_ph(q, vec, sum[v]);
                }
            }

            // Reduce and store
    #pragma unroll
            for (int32_t v = 0; v < 16; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // TAIL LOOP: fewer than 16 vectors
        for (; processedCount < numVectors; ++processedCount) {
            const _Float16* vec =
                reinterpret_cast<const _Float16*>(
                    srchContext->getVectorPointer(&internalVectorIds[processedCount]));

            __m512 sum = _mm512_setzero_ps();

            for (int32_t i = 0; i < mainDim; i += 64) {
                __m512h q0 = _mm512_load_ph(queryPtr + i);
                __m512h q1 = _mm512_load_ph(queryPtr + i + 32);

                __m512h v0 = _mm512_loadu_ph(vec + i);
                __m512h v1 = _mm512_loadu_ph(vec + i + 32);

                sum = _mm512_fmadd_ph(q0, v0, sum);
                sum = _mm512_fmadd_ph(q1, v1, sum);
            }

            if (mainDim < dim) {
                const int32_t rem = dim - mainDim;
                const __mmask32 mask = (1u << rem) - 1;

                __m512h q = _mm512_maskz_load_ph(mask, queryPtr + mainDim);
                __m512h v = _mm512_maskz_loadu_ph(mask, vec + mainDim);
                sum = _mm512_fmadd_ph(q, v, sum);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sum);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) final {
        int32_t processedCount = 0;
        const _Float16* vectors[16];
        const _Float16* queryPtr =
            reinterpret_cast<const _Float16*>(srchContext->queryVectorSimdAligned);

        const int32_t dim = srchContext->dimension;
        const int32_t mainDim = dim & ~63;  // multiple of 64

        __builtin_prefetch(queryPtr, 0, 3);

        // MAIN LOOP — 16 vectors
        for (; processedCount <= numVectors - 16; processedCount += 16) {
            srchContext->getVectorPointersInBulk(
                reinterpret_cast<uint8_t**>(vectors), &internalVectorIds[processedCount], 16);

            __m512 sum[16];
    #pragma unroll
            for (int32_t v = 0; v < 16; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Dimension loop: 64 FP16 per iteration
            for (int32_t i = 0; i < mainDim; i += 64) {
                __m512h q0 = _mm512_load_ph(queryPtr + i);
                __m512h q1 = _mm512_load_ph(queryPtr + i + 32);

    #pragma unroll
                for (int32_t v = 0; v < 16; ++v) {
                    const _Float16* vec = vectors[v];

                    __m512h v0 = _mm512_loadu_ph(vec + i);
                    __m512h v1 = _mm512_loadu_ph(vec + i + 32);

                    // diff = q - v
                    __m512h d0 = _mm512_sub_ph(q0, v0);
                    __m512h d1 = _mm512_sub_ph(q1, v1);

                    // sum += diff * diff
                    sum[v] = _mm512_fmadd_ph(d0, d0, sum[v]);
                    sum[v] = _mm512_fmadd_ph(d1, d1, sum[v]);
                }
            }

            // Tail dimension (single masked step)
            if (mainDim < dim) {
                const int32_t rem = dim - mainDim;
                const __mmask32 mask = (1u << rem) - 1;

                __m512h q = _mm512_maskz_load_ph(mask, queryPtr + mainDim);

    #pragma unroll
                for (int32_t v = 0; v < 16; ++v) {
                    __m512h vec = _mm512_maskz_loadu_ph(mask, vectors[v] + mainDim);
                    __m512h d = _mm512_sub_ph(q, vec);
                    sum[v] = _mm512_fmadd_ph(d, d, sum[v]);
                }
            }

    #pragma unroll
            for (int32_t v = 0; v < 16; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // TAIL LOOP
        for (; processedCount < numVectors; ++processedCount) {
            const _Float16* vec =
                reinterpret_cast<const _Float16*>(
                    srchContext->getVectorPointer(&internalVectorIds[processedCount]));

            __m512 sum = _mm512_setzero_ps();

            for (int32_t i = 0; i < mainDim; i += 64) {
                __m512h q0 = _mm512_load_ph(queryPtr + i);
                __m512h q1 = _mm512_load_ph(queryPtr + i + 32);

                __m512h v0 = _mm512_loadu_ph(vec + i);
                __m512h v1 = _mm512_loadu_ph(vec + i + 32);

                __m512h d0 = _mm512_sub_ph(q0, v0);
                __m512h d1 = _mm512_sub_ph(q1, v1);

                sum = _mm512_fmadd_ph(d0, d0, sum);
                sum = _mm512_fmadd_ph(d1, d1, sum);
            }

            if (mainDim < dim) {
                const int32_t rem = dim - mainDim;
                const __mmask32 mask = (1u << rem) - 1;

                __m512h q = _mm512_maskz_load_ph(mask, queryPtr + mainDim);
                __m512h v = _mm512_maskz_loadu_ph(mask, vec + mainDim);
                __m512h d = _mm512_sub_ph(q, v);
                sum = _mm512_fmadd_ph(d, d, sum);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sum);
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
