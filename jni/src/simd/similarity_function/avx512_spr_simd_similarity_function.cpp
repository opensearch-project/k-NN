#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"


// BF16 -> FP32: zero-extend 16 -> 32 bits then shift left by 16
static inline __m512 cvtbf16_ps(__m256i bf16x16) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16x16), 16));
}

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

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim = dim - simdDim;

        // Precompute tail mask
        const __mmask16 tailMask = tailDim > 0 ? (__mmask16)((1U << tailDim) - 1) : 0;

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

            // A no-mask hot-loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                // Convert N FP16 values to FP32 values per each vector.
                // vRegs[i] will hold N FP32 converted values from ith vector.
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
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

            // Single masked tail
            if (tailDim > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                }

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

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            if (tailDim > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
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

        // Use 8 to keep the register pressure low
        constexpr int32_t vecBlock = 8;
        // Maximum number of elements to load at the same time
        constexpr int32_t elemPerLoad = 16;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim   = dim - simdDim;

        // Precompute tail mask
        const __mmask16 tailMask = tailDim > 0 ? (__mmask16)((1U << tailDim) - 1) : 0;

        // L2 partial sum tracking per each vector
        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            // Init sum variables
            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                // Load queries
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                // Convert N FP16 values to FP32 values per each vector.
                // vRegs[i] will hold N FP32 converted values from ith vector.
                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
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

                // L2 MATH: (q - v)^2 + sum
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    // Compute difference: diff = q - v
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    // Square and Accumulate: sum = (diff * diff) + sum
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
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
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                // Have N FP32 values from query
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                // Have N FP32 values from vector
                __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                // Do FMA e.g. L2 = L2 + diff * diff
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            if (tailDim > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            // __m512 have 16 FP32 values.
            // __m512_reduce_add_ps is summing the values stored in __m512.
            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        // Now, convert score values to L2 score scheme that Lucene uses.
        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
//  BF16 — Uses Native AVX512-BF16 instructions
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

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 32;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim = dim - simdDim;

        // Precompute tail mask
        const __mmask32 tailMask = tailDim > 0 ? (__mmask32)((1ULL << tailDim) - 1) : 0;

        // Preconvert fp32 query to bf16 once
        const int32_t queryBF16Len = ((dim + 31) / 32) * 32;
        uint16_t queryBF16[queryBF16Len] __attribute__((aligned(64)));

        int32_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256i bf = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(queryPtr + i));
            _mm256_storeu_si256((__m256i*)(queryBF16 + i), bf);
        }
        if (i < dim) {
            __mmask16 mask = (__mmask16)((1U << (dim - i)) - 1);
            __m256i bf = (__m256i)_mm512_cvtneps_pbh(_mm512_maskz_loadu_ps(mask, queryPtr + i));
            _mm256_mask_storeu_epi16(queryBF16 + i, mask, bf);
        }
        for (int32_t j = dim; j < queryBF16Len; ++j) {
            queryBF16[j] = 0;
        }

        __m512 sum[vecBlock];

        // Bulk loop
        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop — single 64B bf16 query load, no conversion
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512bh q = (__m512bh)_mm512_loadu_si512(queryBF16 + i);

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextOffset = i + elemPerLoad;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextOffset * 2, 0, 3);
                    }
                    __builtin_prefetch(queryBF16 + nextOffset, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512bh vec = (__m512bh)_mm512_loadu_si512(vectors[v] + 2 * i);
                    sum[v] = _mm512_dpbf16_ps(sum[v], q, vec);
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m512bh q = (__m512bh)_mm512_maskz_loadu_epi16(tailMask, queryBF16 + simdDim);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512bh vec = (__m512bh)_mm512_maskz_loadu_epi16(
                        tailMask, vectors[v] + 2 * simdDim);
                    sum[v] = _mm512_dpbf16_ps(sum[v], q, vec);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 s = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                s = _mm512_dpbf16_ps(s,
                        (__m512bh)_mm512_loadu_si512(queryBF16 + i),
                        (__m512bh)_mm512_loadu_si512(vecPtr + 2 * i));
            }

            if (tailDim > 0) {
                __m512bh q = (__m512bh)_mm512_maskz_loadu_epi16(tailMask, queryBF16 + simdDim);
                s = _mm512_dpbf16_ps(s, q,
                    (__m512bh)_mm512_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
            }

            scores[processedCount] = _mm512_reduce_add_ps(s);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
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
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]); }
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
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
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
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            if (tailDim > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                __m512 v = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// SQ (ADC: 4-bit query x 1-bit data) - AVX512 SIMD implementation
//
// The query is 4-bit quantized and transposed into 4 bit planes (via transposeHalfByte).
// Each bit plane has `binaryCodeBytes` bytes. The int4BitDotProduct computes:
//   Result = popcount(plane0 AND data) * 1
//          + popcount(plane1 AND data) * 2
//          + popcount(plane2 AND data) * 4
//          + popcount(plane3 AND data) * 8
//

static constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

// Reads the per-vector correction factors from a potentially unaligned address.
// On-disk layout after binaryCode: [lowerInterval(f32)][upperInterval(f32)][additionalCorrection(f32)][quantizedComponentSum(i32)]
static FORCE_INLINE void readDataCorrections(const uint8_t* ptr, float& ax, float& lx, float& additional, float& x1) {
    float lower, upper;
    std::memcpy(&lower,      ptr,      sizeof(float));
    std::memcpy(&upper,      ptr + 4,  sizeof(float));
    std::memcpy(&additional, ptr + 8,  sizeof(float));
    int32_t componentSum;
    std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
    ax = lower;
    lx = upper - lower;
    x1 = static_cast<float>(componentSum);
}

// Scalar fallback for int4BitDotProduct
static FORCE_INLINE int64_t int4BitDotProduct(const uint8_t* q, const uint8_t* d, const int32_t binaryCodeBytes) {
    int64_t result = 0;
    for (int32_t bitPlane = 0 ; bitPlane < 4 ; ++bitPlane) {
        const int32_t words = binaryCodeBytes >> 3;

        int64_t subResult = 0;
        for (int32_t w = 0 ; w < words ; ++w) {
            uint64_t qWord, dWord;
            std::memcpy(&qWord, q + bitPlane * binaryCodeBytes + w * 8, sizeof(uint64_t));
            std::memcpy(&dWord, d + w * 8, sizeof(uint64_t));
            subResult += __builtin_popcountll(qWord & dWord);
        }

        const int32_t remainStart = words * 8;
        for (int32_t r = remainStart ; r < binaryCodeBytes ; ++r) {
            subResult += __builtin_popcount((q[bitPlane * binaryCodeBytes + r] & d[r]) & 0xFF);
        }

        result += subResult << bitPlane;
    }
    return result;
}

// AVX512 per-byte popcount using nibble LUT
static FORCE_INLINE __m512i avx512_popcnt_epi8(const __m512i v) {
    alignas(64) static const __m512i popLut = _mm512_setr_epi64(
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL);
    const __m512i lowMask = _mm512_set1_epi8(0x0F);

    __m512i lo = _mm512_and_si512(v, lowMask);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask);
    __m512i cntLo = _mm512_shuffle_epi8(popLut, lo);
    __m512i cntHi = _mm512_shuffle_epi8(popLut, hi);
    return _mm512_add_epi8(cntLo, cntHi);
}

// AVX512 SIMD batched int4BitDotProduct
template <int BATCH_SIZE>
static FORCE_INLINE void avx512_4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t binaryCodeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;

    __m512i acc[BATCH_SIZE];
    #pragma unroll
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        acc[b] = _mm512_setzero_si512();
    }

    int32_t i = 0;
    for ( ; i + 64 <= binaryCodeBytes ; i += 64) {
        __m512i q0 = _mm512_loadu_si512(plane0 + i);
        __m512i q1 = _mm512_loadu_si512(plane1 + i);
        __m512i q2 = _mm512_loadu_si512(plane2 + i);
        __m512i q3 = _mm512_loadu_si512(plane3 + i);

        if (i + 64 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 64);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 64);
            }
        }

        #pragma unroll
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            __m512i d = _mm512_loadu_si512(dataVecs[b] + i);

            __m512i p0 = avx512_popcnt_epi8(_mm512_and_si512(q0, d));
            __m512i p1 = avx512_popcnt_epi8(_mm512_and_si512(q1, d));
            __m512i p2 = avx512_popcnt_epi8(_mm512_and_si512(q2, d));
            __m512i p3 = avx512_popcnt_epi8(_mm512_and_si512(q3, d));

            __m512i weighted = _mm512_add_epi8(p0, _mm512_slli_epi16(p1, 1));
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p2, 2));
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p3, 3));

            __m512i sad = _mm512_sad_epu8(weighted, _mm512_setzero_si512());
            acc[b] = _mm512_add_epi64(acc[b], sad);
        }
    }

    #pragma unroll
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(_mm512_reduce_add_epi64(acc[b]));
    }

    for ( ; i < binaryCodeBytes ; ++i) {
        uint8_t q0b = plane0[i], q1b = plane1[i], q2b = plane2[i], q3b = plane3[i];
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            uint8_t db = dataVecs[b][i];
            results[b] += static_cast<float>(
                __builtin_popcount((q0b & db) & 0xFF) * 1
              + __builtin_popcount((q1b & db) & 0xFF) * 2
              + __builtin_popcount((q2b & db) & 0xFF) * 4
              + __builtin_popcount((q3b & db) & 0xFF) * 8);
        }
    }
}

template <bool IsMaxIP>
struct AVX512SQSimilarityFunction final : SimilarityFunction {
    HOT_SPOT void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                            int32_t* internalVectorIds,
                                            float* scores,
                                            const int32_t numVectors) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t binaryCodeBytes = (dim + 7) / 8;

        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw; std::memcpy(&y1Raw, &queryCorrectionPtr[3], sizeof(int32_t));
        const float y1 = static_cast<float>(y1Raw);
        const float centroidDp = queryCorrectionPtr[4];

        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 8;
        constexpr int32_t vecHalfBlock = 4;
        uint8_t* vectors[vecBlock];

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);
            avx512_4bitDotProductBatch<vecBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            #pragma unroll
            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                if ((i + 1) < vecBlock) {
                    __builtin_prefetch(vectors[i + 1] + binaryCodeBytes);
                }
                float ax, lx, additional, x1;
                readDataCorrections(vectors[i] + binaryCodeBytes, ax, lx, additional, x1);

                scores[processedCount + i] = ax * ay * dim
                                           + ay * lx * x1
                                           + ax * ly * y1
                                           + lx * ly * scores[processedCount + i];

                if constexpr (IsMaxIP) {
                    scores[processedCount + i] += queryAdditional + additional - centroidDp;
                } else {
                    scores[processedCount + i] = std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount + i]);
                }
            }
        }

        for ( ; (processedCount + vecHalfBlock) <= numVectors ; processedCount += vecHalfBlock) {
            srchContext->getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecHalfBlock);
            avx512_4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            #pragma unroll
            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
                if ((i + 1) < vecHalfBlock) {
                    __builtin_prefetch(vectors[i + 1] + binaryCodeBytes);
                }
                float ax, lx, additional, x1;
                readDataCorrections(vectors[i] + binaryCodeBytes, ax, lx, additional, x1);

                scores[processedCount + i] = ax * ay * dim
                                           + ay * lx * x1
                                           + ax * ly * y1
                                           + lx * ly * scores[processedCount + i];

                if constexpr (IsMaxIP) {
                    scores[processedCount + i] += queryAdditional + additional - centroidDp;
                } else {
                    scores[processedCount + i] =
                        std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount + i]);
                }
            }
        }

        for ( ; processedCount < numVectors ; ++processedCount) {
            const auto* dataVec = srchContext->getVectorPointer(internalVectorIds[processedCount]);
            const float qcDist = static_cast<float>(
                int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

            float ax, lx, additional, x1;
            readDataCorrections(dataVec + binaryCodeBytes, ax, lx, additional, x1);

            scores[processedCount] = ax * ay * dim
                                   + ay * lx * x1
                                   + ax * ly * y1
                                   + lx * ly * qcDist;

            if constexpr (IsMaxIP) {
                scores[processedCount] += queryAdditional + additional - centroidDp;
            } else {
                scores[processedCount] =
                    std::max(0.0F, queryAdditional + additional - 2 * scores[processedCount]);
            }
        }

        if constexpr (IsMaxIP) {
            FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, numVectors);
        } else {
            FaissScoreToLuceneScoreTransform::l2TransformBulk(scores, numVectors);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t dim = srchContext->dimension;
        const int32_t binaryCodeBytes = (dim + 7) / 8;

        const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
        const float ay = queryCorrectionPtr[0];
        const float ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
        const float queryAdditional = queryCorrectionPtr[2];
        int32_t y1Raw2; std::memcpy(&y1Raw2, &queryCorrectionPtr[3], sizeof(int32_t));
        const float y1 = static_cast<float>(y1Raw2);
        const float centroidDp = queryCorrectionPtr[4];

        const auto* dataVec = srchContext->getVectorPointer(internalVectorId);
        const float qcDist = static_cast<float>(
            int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

        float ax, lx, additional, x1;
        readDataCorrections(dataVec + binaryCodeBytes, ax, lx, additional, x1);

        float score = ax * ay * dim
                      + ay * lx * x1
                      + ax * ly * y1
                      + lx * ly * qcDist;

        if constexpr (IsMaxIP) {
            score += queryAdditional + additional - centroidDp;
            return FaissScoreToLuceneScoreTransform::ipToMaxIpTransform(score);
        } else {
            score = std::max(0.0F, queryAdditional + additional - 2 * score);
            return FaissScoreToLuceneScoreTransform::l2Transform(score);
        }
    }
};

//
// FP16
//
// 1. Max IP
AVX512SPRFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512SPRFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;

//
// BF16
//
// 1. Max IP
AVX512BF16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512BF16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> BF16_L2_SIMIL_FUNC;

//
// SQ
//
// 1. Max IP
AVX512SQSimilarityFunction<true> SQ_IP_SIMIL_FUNC;
// 2. L2
AVX512SQSimilarityFunction<false> SQ_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_IP) {
        return &SQ_IP_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_L2) {
        return &SQ_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT) {
        return &BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_L2) {
        return &BF16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
