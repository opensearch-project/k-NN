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

// AVX512-SPR build. BF16 inner product uses native AVX512-BF16 instructions
// (_mm512_dpbf16_ps / _mm512_cvtneps_pbh) for maximum throughput.
#include "avx512_common_simd_similarity_function.cpp"

//
//  BF16 — Uses Native AVX512-BF16 instructions
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    // Scores `internalVectorIds[0..numVectors)` into `scores` against `queryBF16`, the query already converted to
    // BF16 and zero-padded to a multiple of 32 elements (see calculateSimilarityInBulk). Format (bf16_format.h)
    // locates each vector's BF16 values; Epilogue turns the raw inner products into scores in place. Both are
    // compile-time parameters.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         const uint16_t* queryBF16,
                                         const int32_t dim,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_bf16_format<Format>::value, "Format must be one of the bf16_format.h formats");

        int32_t processedCount = 0;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 32;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim = dim - simdDim;

        // Precompute tail mask
        const __mmask32 tailMask = tailDim > 0 ? (__mmask32)((1ULL << tailDim) - 1) : 0;

        __m512 sum[vecBlock];

        // Bulk loop
        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            uint8_t* vectors[vecBlock];
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);

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
            const auto* vecPtr = (const uint8_t*) format.getVectorPointer(internalVectorIds[processedCount]);
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

        epilogue.finishInBulk(scores, numVectors);
    }

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        // Pre-convert the fp32 query to bf16 once per call into the dedicated, reused query buffer.
        // Using srchContext->queryConvertedBuffer (grown, never shrunk) avoids a heap allocation on
        // every bulk-scoring call. It is kept separate from tmpBuffer because getVectorPointer()
        // may resize/reallocate tmpBuffer mid-scan when reassembling a vector that straddles two
        // mmap regions, which would dangle this query pointer. The buffer is sized in bytes; all
        // accesses in the kernel use unaligned loads, so no explicit over-alignment is required.
        const int32_t queryBF16Len = ((dim + 31) / 32) * 32;
        const size_t queryBF16Bytes = static_cast<size_t>(queryBF16Len) * sizeof(uint16_t);
        auto& queryBuf = srchContext->queryConvertedBuffer;
        if (queryBuf.size() < queryBF16Bytes) {
            queryBuf.resize(queryBF16Bytes);
        }
        uint16_t* queryBF16 = reinterpret_cast<uint16_t*>(queryBuf.data());

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

        // Lucene's flat BF16 records; the transform is the one this function was instantiated with.
        const LuceneBf16Format format(srchContext);
        const LuceneBf16Format::Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, queryBF16, dim, internalVectorIds, scores, numVectors);
    }
};

#include "avx512_common_registration.cpp"
