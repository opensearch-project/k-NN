#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
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
        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 16;

        const int32_t simdDim = dim & ~(elemPerLoad - 1);
        const int32_t tailDim = dim & (elemPerLoad - 1);
        const int32_t simdDim2 = simdDim & ~(2 * elemPerLoad - 1);

        constexpr int32_t kPrefetchAheadElems = 8 * elemPerLoad;

        // For next-batch prefetch: compute vector addresses inline to
        // warm TLB and seed HW prefetcher during current batch computation.
        const auto* mmapBase = (srchContext->mmapPages.size() == 1)
            ? reinterpret_cast<const uint8_t*>(srchContext->mmapPages[0])
            : nullptr;
        const int64_t vecByteStride = srchContext->oneVectorByteSize;

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* dataPtrs[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)dataPtrs, &internalVectorIds[processedCount], vecBlock);

            // Warm first 3 cache lines of each vector into L1/L2.
            // The first prefetch (L1) also triggers a TLB walk if needed.
            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                __builtin_prefetch(dataPtrs[v], 0, 3);
                __builtin_prefetch(dataPtrs[v] + 64, 0, 2);
                __builtin_prefetch(dataPtrs[v] + 128, 0, 2);
            }

            // Pipeline: prefetch NEXT batch's first cache lines to overlap
            // TLB page walks (~140 cycles each) and DRAM fetches (~200+ cycles)
            // with current batch computation (~4000+ cycles at dim=768).
            // Uses inline address computation to avoid getVectorPointersInBulk
            // side effects on tmpBuffer.
            const int32_t nextStart = processedCount + vecBlock;
            if (mmapBase && nextStart <= numVectors - vecBlock) {
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    const auto* p = mmapBase + vecByteStride * internalVectorIds[nextStart + v];
                    __builtin_prefetch(p, 0, 2);
                    __builtin_prefetch(p + 64, 0, 2);
                    __builtin_prefetch(p + 128, 0, 2);
                }
            }

            __m512 ipAccum0[vecBlock];
            __m512 ipAccum1[vecBlock];

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                ipAccum0[v] = _mm512_setzero_ps();
                ipAccum1[v] = _mm512_setzero_ps();
            }

            // 2x unrolled hot loop with L2-targeted prefetch (prefetcht1).
            // Unlike prefetcht0, this routes through the superqueue (48 entries)
            // instead of L1D fill buffers (12 entries), directly eliminating
            // the 30% fb_full stall from the profile. Demand loads then hit
            // in L2 (~10 cycles), fully hidden by OoO across 8 independent
            // accumulator chains.
            int32_t i = 0;
            for (; i < simdDim2; i += 2 * elemPerLoad) {
                __m512 queryChunk0 = _mm512_loadu_ps(queryPtr + i);
                __m512 queryChunk1 = _mm512_loadu_ps(queryPtr + i + elemPerLoad);

                if ((i + kPrefetchAheadElems) < dim) {
                    const int32_t prefOffset = (i + kPrefetchAheadElems) * 2;

                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(dataPtrs[v] + prefOffset, 0, 2);
                    }
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * i)));
                    __m512 dataChunk1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * (i + elemPerLoad))));
                    ipAccum0[v] = _mm512_fmadd_ps(queryChunk0, dataChunk0, ipAccum0[v]);
                    ipAccum1[v] = _mm512_fmadd_ps(queryChunk1, dataChunk1, ipAccum1[v]);
                }
            }

            for (; i < simdDim; i += elemPerLoad) {
                __m512 queryChunk = _mm512_loadu_ps(queryPtr + i);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * i)));
                    ipAccum0[v] = _mm512_fmadd_ps(queryChunk, dataChunk, ipAccum0[v]);
                }
            }

            if (tailDim > 0) {
                const __mmask16 tailMask = (__mmask16)(1U << tailDim) - 1;
                __m512 queryChunk = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, dataPtrs[v] + 2 * simdDim));
                    ipAccum0[v] = _mm512_fmadd_ps(queryChunk, dataChunk, ipAccum0[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(_mm512_add_ps(ipAccum0[v], ipAccum1[v]));
            }
        }

        // Scalar tail with one-vector-ahead prefetch pipeline
        {
            constexpr int32_t unrollFactor = 4;

            for (; processedCount < numVectors; ++processedCount) {
                const auto* dataPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);

                if (processedCount + 1 < numVectors) {
                    const auto* nextPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount + 1]);
                    __builtin_prefetch(nextPtr, 0, 3);
                    __builtin_prefetch(nextPtr + 64, 0, 2);
                }

                __m512 ipPartialAccum[unrollFactor];

                #pragma unroll
                for (int32_t u = 0; u < unrollFactor; ++u) {
                    ipPartialAccum[u] = _mm512_setzero_ps();
                }

                const int32_t unrolledDim = simdDim & ~(unrollFactor * elemPerLoad - 1);
                int32_t i = 0;

                for (; i < unrolledDim; i += unrollFactor * elemPerLoad) {
                    #pragma unroll
                    for (int32_t u = 0; u < unrollFactor; ++u) {
                        __m512 queryChunk = _mm512_loadu_ps(queryPtr + i + u * elemPerLoad);
                        __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtr + 2 * (i + u * elemPerLoad))));
                        ipPartialAccum[u] = _mm512_fmadd_ps(queryChunk, dataChunk, ipPartialAccum[u]);
                    }
                }

                for (; i < simdDim; i += elemPerLoad) {
                    __m512 queryChunk = _mm512_loadu_ps(queryPtr + i);
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtr + 2 * i)));
                    ipPartialAccum[0] = _mm512_fmadd_ps(queryChunk, dataChunk, ipPartialAccum[0]);
                }

                if (tailDim > 0) {
                    const __mmask16 tailMask = (__mmask16)((1U << tailDim) - 1);
                    __m512 queryChunk = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, dataPtr + 2 * simdDim));
                    ipPartialAccum[0] = _mm512_fmadd_ps(queryChunk, dataChunk, ipPartialAccum[0]);
                }

                scores[processedCount] = _mm512_reduce_add_ps(
                    _mm512_add_ps(_mm512_add_ps(ipPartialAccum[0], ipPartialAccum[1]),
                                  _mm512_add_ps(ipPartialAccum[2], ipPartialAccum[3])));
            }
        }

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
        // == (dim / elemPerLoad) * elemPerLoad
        const int32_t simdDim = dim & ~(elemPerLoad - 1);
        // == dim % elemPerLoad
        const int32_t tailDim = dim & (elemPerLoad - 1);
        // 2x unrolled boundary
        // == (simdDim / (2 * elemPerLoad)) * (2 * elemPerLoad)
        const int32_t simdDim2 = simdDim & ~(2 * elemPerLoad - 1);

        constexpr int32_t kPrefetchAheadElems = 8 * elemPerLoad;

        const auto* mmapBase = (srchContext->mmapPages.size() == 1)
            ? reinterpret_cast<const uint8_t*>(srchContext->mmapPages[0])
            : nullptr;
        const int64_t vecByteStride = srchContext->oneVectorByteSize;

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* dataPtrs[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)dataPtrs, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                __builtin_prefetch(dataPtrs[v], 0, 3);
                __builtin_prefetch(dataPtrs[v] + 64, 0, 2);
                __builtin_prefetch(dataPtrs[v] + 128, 0, 2);
            }

            const int32_t nextStart = processedCount + vecBlock;
            if (mmapBase && nextStart <= numVectors - vecBlock) {
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    const auto* p = mmapBase + vecByteStride * internalVectorIds[nextStart + v];
                    __builtin_prefetch(p, 0, 2);
                    __builtin_prefetch(p + 64, 0, 2);
                    __builtin_prefetch(p + 128, 0, 2);
                }
            }

            __m512 sqDistAccum0[vecBlock];
            __m512 sqDistAccum1[vecBlock];

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sqDistAccum0[v] = _mm512_setzero_ps();
                sqDistAccum1[v] = _mm512_setzero_ps();
            }

            int32_t i = 0;
            for (; i < simdDim2; i += 2 * elemPerLoad) {
                __m512 queryChunk0 = _mm512_loadu_ps(queryPtr + i);
                __m512 queryChunk1 = _mm512_loadu_ps(queryPtr + i + elemPerLoad);

                if ((i + kPrefetchAheadElems) < dim) {
                    const int32_t prefOffset = (i + kPrefetchAheadElems) * 2;

                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(dataPtrs[v] + prefOffset, 0, 2);
                    }
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * i)));
                    __m512 dataChunk1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * (i + elemPerLoad))));
                    __m512 residual0 = _mm512_sub_ps(queryChunk0, dataChunk0);
                    __m512 residual1 = _mm512_sub_ps(queryChunk1, dataChunk1);
                    sqDistAccum0[v] = _mm512_fmadd_ps(residual0, residual0, sqDistAccum0[v]);
                    sqDistAccum1[v] = _mm512_fmadd_ps(residual1, residual1, sqDistAccum1[v]);
                }
            }

            for (; i < simdDim; i += elemPerLoad) {
                __m512 queryChunk = _mm512_loadu_ps(queryPtr + i);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtrs[v] + 2 * i)));
                    __m512 residual = _mm512_sub_ps(queryChunk, dataChunk);
                    sqDistAccum0[v] = _mm512_fmadd_ps(residual, residual, sqDistAccum0[v]);
                }
            }

            if (tailDim > 0) {
                const __mmask16 tailMask = (__mmask16)(1U << tailDim) - 1;
                __m512 queryChunk = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, dataPtrs[v] + 2 * simdDim));
                    __m512 residual = _mm512_sub_ps(queryChunk, dataChunk);
                    sqDistAccum0[v] = _mm512_fmadd_ps(residual, residual, sqDistAccum0[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(_mm512_add_ps(sqDistAccum0[v], sqDistAccum1[v]));
            }
        }

        // Scalar tail with one-vector-ahead prefetch pipeline
        {
            constexpr int32_t unrollFactor = 4;

            for (; processedCount < numVectors; ++processedCount) {
                const auto* dataPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);

                if (processedCount + 1 < numVectors) {
                    const auto* nextPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount + 1]);
                    __builtin_prefetch(nextPtr, 0, 3);
                    __builtin_prefetch(nextPtr + 64, 0, 2);
                }

                __m512 sqDistPartialAccum[unrollFactor];

                #pragma unroll
                for (int32_t u = 0; u < unrollFactor; ++u) {
                    sqDistPartialAccum[u] = _mm512_setzero_ps();
                }

                const int32_t unrolledDim = simdDim & ~(unrollFactor * elemPerLoad - 1);
                int32_t i = 0;

                for (; i < unrolledDim; i += unrollFactor * elemPerLoad) {
                    #pragma unroll
                    for (int32_t u = 0; u < unrollFactor; ++u) {
                        __m512 queryChunk = _mm512_loadu_ps(queryPtr + i + u * elemPerLoad);
                        __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtr + 2 * (i + u * elemPerLoad))));
                        __m512 residual = _mm512_sub_ps(queryChunk, dataChunk);
                        sqDistPartialAccum[u] = _mm512_fmadd_ps(residual, residual, sqDistPartialAccum[u]);
                    }
                }

                for (; i < simdDim; i += elemPerLoad) {
                    __m512 queryChunk = _mm512_loadu_ps(queryPtr + i);
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(dataPtr + 2 * i)));
                    __m512 residual = _mm512_sub_ps(queryChunk, dataChunk);
                    sqDistPartialAccum[0] = _mm512_fmadd_ps(residual, residual, sqDistPartialAccum[0]);
                }

                if (tailDim > 0) {
                    const __mmask16 tailMask = (__mmask16)((1U << tailDim) - 1);
                    __m512 queryChunk = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                    __m512 dataChunk = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, dataPtr + 2 * simdDim));
                    __m512 residual = _mm512_sub_ps(queryChunk, dataChunk);
                    sqDistPartialAccum[0] = _mm512_fmadd_ps(residual, residual, sqDistPartialAccum[0]);
                }

                scores[processedCount] = _mm512_reduce_add_ps(
                    _mm512_add_ps(_mm512_add_ps(sqDistPartialAccum[0], sqDistPartialAccum[1]),
                                  _mm512_add_ps(sqDistPartialAccum[2], sqDistPartialAccum[3])));
            }
        }

        // Now, convert score values to L2 score scheme that Lucene uses.
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
// Because oneVectorByteSize may not be a multiple of 4, subsequent vectors can start at
// non-4-byte-aligned offsets, making reinterpret_cast<float*> undefined behaviour.
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
// q has 4 * binaryCodeBytes bytes (4 bit planes), d has binaryCodeBytes bytes
// Uses std::memcpy for uint64_t loads to avoid undefined behavior from unaligned
// reinterpret_cast when binaryCodeBytes is not a multiple of 8. Compilers optimize
// the 8-byte memcpy into a single mov instruction — zero runtime cost.
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

// AVX512 per-byte popcount using nibble LUT (works on all AVX512F/BW targets).
// Uses vpshufb with a 4-bit lookup table to count bits in each byte of a 512-bit register.
static FORCE_INLINE __m512i avx512_popcnt_epi8(const __m512i v) {
    // Nibble popcount lookup table: {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4} replicated across all 64-byte lanes
    // index : value : popcount
    //   0     : 0000  : 0
    //   1     : 0001  : 1
    //   2     : 0010  : 1
    //   3     : 0011  : 2
    //   ...
    //   15    : 1111  : 4
    // Example:
    // 0x0403030203020201LL
    // Split it, we get:
    // 0x04 03 03 02 03 02 02 01
    //        index 9 -----^  ^----- index 8
    // Witch maps to
    // LUT[8]  = 1 -> 01b, the last value
    // LUT[9]  = 2 -> 02b, the second value from right
    // LUT[10] = 2
    // LUT[11] = 3
    // LUT[12] = 2
    // LUT[13] = 3
    // LUT[14] = 3
    // LUT[15] = 4 -> the first value 0x04
    alignas(64) static const __m512i popLut = _mm512_setr_epi64(
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL,
        0x0302020102010100LL, 0x0403030203020201LL);
    const __m512i lowMask = _mm512_set1_epi8(0x0F);

    // Split each byte into low and high nibbles, look up popcount for each, sum
    // Example:
    // v = 0b10110110
    // Split:
    //   hi = 1011 (11) → popcount = 3
    //   lo = 0110 (6)  → popcount = 2
    // Instead of popcount, we can do table look-up, and we get:
    // LUT[11] = 3
    // LUT[6]  = 2
    // 3 + 2 = 5 = popcount(10110110)
    __m512i lo = _mm512_and_si512(v, lowMask);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask);
    __m512i cntLo = _mm512_shuffle_epi8(popLut, lo);
    __m512i cntHi = _mm512_shuffle_epi8(popLut, hi);
    return _mm512_add_epi8(cntLo, cntHi);
}

// AVX512 SIMD batched int4BitDotProduct.
// Processes 64 bytes per iteration
// Uses LUT-based per-byte popcount on each plane, then weights by 1/2/4/8.
template <int BATCH_SIZE>
static FORCE_INLINE void avx512_4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t binaryCodeBytes,
    float* results) {

    // Query vector is transposed
    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;

    // 64-bit accumulators to avoid overflow (each iteration can add up to 64*120 = 7680 per 64-byte chunk)
    __m512i acc[BATCH_SIZE];
    #pragma unroll
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        acc[b] = _mm512_setzero_si512();
    }

    int32_t i = 0;
    for ( ; i + 64 <= binaryCodeBytes ; i += 64) {
        // Load 64 bytes from each query plane (shared across all data vectors)
        __m512i q0 = _mm512_loadu_si512(plane0 + i);
        __m512i q1 = _mm512_loadu_si512(plane1 + i);
        __m512i q2 = _mm512_loadu_si512(plane2 + i);
        __m512i q3 = _mm512_loadu_si512(plane3 + i);

        // Prefetch next chunk
        if (i + 64 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 64);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 64);
            }
        }

        #pragma unroll
        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            // Load 64 bytes of data vector's binary code
            __m512i d = _mm512_loadu_si512(dataVecs[b] + i);

            // AND each plane with data, then per-byte popcount
            __m512i p0 = avx512_popcnt_epi8(_mm512_and_si512(q0, d));
            __m512i p1 = avx512_popcnt_epi8(_mm512_and_si512(q1, d));
            __m512i p2 = avx512_popcnt_epi8(_mm512_and_si512(q2, d));
            __m512i p3 = avx512_popcnt_epi8(_mm512_and_si512(q3, d));

            // Weight: p0*1 + p1*2 + p2*4 + p3*8
            // Max per byte: 8*1 + 8*2 + 8*4 + 8*8 = 120, fits in uint8_t
            // Note: _mm512_slli_epi16 shifts 16-bit lanes, but since popcount values are at most 8 (0b00001000),
            // shifting left by 1/2/3 won't cause cross-byte bleed within 16-bit lanes (high bits of low byte are 0).
            __m512i weighted = _mm512_add_epi8(p0, _mm512_slli_epi16(p1, 1)); // -> weighted += p2 << 1
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p2, 2)); // -> weighted += p2 << 2
            weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p3, 3)); // -> weighted += p2 << 3

            // Horizontal sum: u8 -> u64 via _mm512_sad_epu8 (sum of absolute differences against zero)
            // _mm512_sad_epu8 sums 8 consecutive u8 values into u64 lanes
            // "SAD" : feeling or showing sorrow; unhappy.
            // kidding, SAD = Sum of Absolute Differences i.e. sum(|a[i] - b[i]|)
            // _mm512_sad_epu8(weighted, _mm512_setzero_si512()) -> |weighted[i] - 0| = weighted[i]
            // so it becomes, sum(weighted[i]), just a sum.
            __m512i sad = _mm512_sad_epu8(weighted, _mm512_setzero_si512());

            // Accumulate into 64-bit accumulators
            acc[b] = _mm512_add_epi64(acc[b], sad);
        }
    }

    // Horizontal sum of 64-bit accumulators into results
    #pragma unroll
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(_mm512_reduce_add_epi64(acc[b]));
    }

    // Scalar tail for remaining bytes (< 64)
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

        // Read query correction factors from tmpBuffer
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

        // Batch size 8
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

        // Batch size 4
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

        // Tail: remaining vectors (scalar)
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
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
