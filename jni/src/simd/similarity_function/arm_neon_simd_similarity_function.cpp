/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#endif
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>
#include <iostream>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"
#include "simd/similarity_function/sq_format.h"
#include "simd/similarity_function/fp16_format.h"
#include "simd/similarity_function/bf16_format.h"

using knn_jni::simd::similarity_function::LuceneSqFormat;
using knn_jni::simd::similarity_function::is_sq_format;
using knn_jni::simd::similarity_function::LuceneFp16Format;
using knn_jni::simd::similarity_function::is_fp16_format;
using knn_jni::simd::similarity_function::LuceneBf16Format;
using knn_jni::simd::similarity_function::is_bf16_format;



//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    // Scores `internalVectorIds[0..numVectors)` into `scores`. Format (fp16_format.h) locates each vector's FP16
    // values; Epilogue turns the raw inner products into scores in place. Both are compile-time parameters.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         const float* queryPtr,
                                         const int32_t dim,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_fp16_format<Format>::value, "Format must be one of the fp16_format.h formats");

        // Bulk inner product with 4 batch
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        uint8_t* vectors[vecBlock];
        constexpr int32_t dimensionBatch = 8;

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);

            // Score accumulator per each vector
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            // Batch inner product for 8 values
            int32_t i = 0;
            for (; i + dimensionBatch <= dim; i += dimensionBatch) {
                // Load 8 FP32 query elements
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);

                // Load 8 FP16 elements from each target and convert to FP32
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vectors[0] + i * 2));
                float16x8_t h1 = vld1q_f16((const __fp16 *)(vectors[1] + i * 2));
                float16x8_t h2 = vld1q_f16((const __fp16 *)(vectors[2] + i * 2));
                float16x8_t h3 = vld1q_f16((const __fp16 *)(vectors[3] + i * 2));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
                float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));
                float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
                float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));
                float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
                float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

                // Post-load prefetch: next 8 elements
                // By the time in the next loop,
                if (i + dimensionBatch < dim) {
                    __builtin_prefetch(queryPtr + i + 8);
                    __builtin_prefetch(vectors[0] + (i + 8) * 2);
                    __builtin_prefetch(vectors[1] + (i + 8) * 2);
                    __builtin_prefetch(vectors[2] + (i + 8) * 2);
                    __builtin_prefetch(vectors[3] + (i + 8) * 2);
                }

                // Accumulate FMA
                acc0 = vfmaq_f32(acc0, q0, d0_lo);
                acc0 = vfmaq_f32(acc0, q1, d0_hi);

                acc1 = vfmaq_f32(acc1, q0, d1_lo);
                acc1 = vfmaq_f32(acc1, q1, d1_hi);

                acc2 = vfmaq_f32(acc2, q0, d2_lo);
                acc2 = vfmaq_f32(acc2, q1, d2_hi);

                acc3 = vfmaq_f32(acc3, q0, d3_lo);
                acc3 = vfmaq_f32(acc3, q1, d3_hi);
            }

            // Horizontal sum
            scores[processedCount] = vaddvq_f32(acc0);
            scores[processedCount + 1] = vaddvq_f32(acc1);
            scores[processedCount + 2] = vaddvq_f32(acc2);
            scores[processedCount + 3] = vaddvq_f32(acc3);

            // Scalar tail.
            // For example,
            // if dimension was 66 then this loop will take care of remaining 2 values.
            for (; i < dim; i++) {
                __fp16 h0 = *((const __fp16 *)(vectors[0] + i * 2));
                __fp16 h1 = *((const __fp16 *)(vectors[1] + i * 2));
                __fp16 h2 = *((const __fp16 *)(vectors[2] + i * 2));
                __fp16 h3 = *((const __fp16 *)(vectors[3] + i * 2));
                const float qv = queryPtr[i];
                scores[processedCount] += qv * (float)h0;
                scores[processedCount + 1] += qv * (float)h1;
                scores[processedCount + 2] += qv * (float)h2;
                scores[processedCount + 3] += qv * (float)h3;
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const __fp16*) format.getVectorPointer(internalVectorIds[processedCount]);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int32_t i = 0;
            for (; i <= dim - dimensionBatch; i += dimensionBatch) {
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vecPtr + i));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                acc = vfmaq_f32(acc, q0, d0_lo);
                acc = vfmaq_f32(acc, q1, d0_hi);
            }

            float finalSum = vaddvq_f32(acc);
            // Scalar tail for dimensions not divisible by 4
            for (; i < dim; ++i) {
                finalSum += queryPtr[i] * (float)vecPtr[i];
            }
            scores[processedCount] = finalSum;
        }

        epilogue.finishInBulk(scores, numVectors);
    }

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        // Lucene's flat FP16 records; the transform is the one this function was instantiated with.
        const LuceneFp16Format format(srchContext);
        const LuceneFp16Format::Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, queryPtr, dim, internalVectorIds, scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    // Scores `internalVectorIds[0..numVectors)` into `scores`. Format (fp16_format.h) locates each vector's FP16
    // values; Epilogue turns the raw squared L2 distances into scores in place. Both are compile-time parameters.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         const float* queryPtr,
                                         const int32_t dim,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_fp16_format<Format>::value, "Format must be one of the fp16_format.h formats");

        // Bulk L2 with 4 batch
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        uint8_t* vectors[vecBlock];
        constexpr int32_t dimensionBatch = 8;

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);

            // Score accumulator per each vector
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            // Batch L2 for 8 values
            int32_t i = 0;
            for (; i + dimensionBatch <= dim; i += dimensionBatch) {
                // Load 8 FP32 query elements
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);

                // Load 8 FP16 elements from each target and convert to FP32
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vectors[0] + i * 2));
                float16x8_t h1 = vld1q_f16((const __fp16 *)(vectors[1] + i * 2));
                float16x8_t h2 = vld1q_f16((const __fp16 *)(vectors[2] + i * 2));
                float16x8_t h3 = vld1q_f16((const __fp16 *)(vectors[3] + i * 2));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                float32x4_t d1_lo = vcvt_f32_f16(vget_low_f16(h1));
                float32x4_t d1_hi = vcvt_f32_f16(vget_high_f16(h1));
                float32x4_t d2_lo = vcvt_f32_f16(vget_low_f16(h2));
                float32x4_t d2_hi = vcvt_f32_f16(vget_high_f16(h2));
                float32x4_t d3_lo = vcvt_f32_f16(vget_low_f16(h3));
                float32x4_t d3_hi = vcvt_f32_f16(vget_high_f16(h3));

                if (i + 32 < dim) {
                    __builtin_prefetch(queryPtr + i + 32);
                    __builtin_prefetch(vectors[0] + (i + 32) * 2);
                    __builtin_prefetch(vectors[1] + (i + 32) * 2);
                    __builtin_prefetch(vectors[2] + (i + 32) * 2);
                    __builtin_prefetch(vectors[3] + (i + 32) * 2);
                }

                // L2: diff = q - d, then acc += diff * diff (one FMA per half)
                float32x4_t diff0_lo = vsubq_f32(q0, d0_lo);
                float32x4_t diff0_hi = vsubq_f32(q1, d0_hi);
                acc0 = vfmaq_f32(acc0, diff0_lo, diff0_lo);
                acc0 = vfmaq_f32(acc0, diff0_hi, diff0_hi);

                float32x4_t diff1_lo = vsubq_f32(q0, d1_lo);
                float32x4_t diff1_hi = vsubq_f32(q1, d1_hi);
                acc1 = vfmaq_f32(acc1, diff1_lo, diff1_lo);
                acc1 = vfmaq_f32(acc1, diff1_hi, diff1_hi);

                float32x4_t diff2_lo = vsubq_f32(q0, d2_lo);
                float32x4_t diff2_hi = vsubq_f32(q1, d2_hi);
                acc2 = vfmaq_f32(acc2, diff2_lo, diff2_lo);
                acc2 = vfmaq_f32(acc2, diff2_hi, diff2_hi);

                float32x4_t diff3_lo = vsubq_f32(q0, d3_lo);
                float32x4_t diff3_hi = vsubq_f32(q1, d3_hi);
                acc3 = vfmaq_f32(acc3, diff3_lo, diff3_lo);
                acc3 = vfmaq_f32(acc3, diff3_hi, diff3_hi);
            }

            // Horizontal sum
            scores[processedCount] = vaddvq_f32(acc0);
            scores[processedCount + 1] = vaddvq_f32(acc1);
            scores[processedCount + 2] = vaddvq_f32(acc2);
            scores[processedCount + 3] = vaddvq_f32(acc3);

            // Scalar tail.
            for (; i < dim; i++) {
                __fp16 h0 = *((const __fp16 *)(vectors[0] + i * 2));
                __fp16 h1 = *((const __fp16 *)(vectors[1] + i * 2));
                __fp16 h2 = *((const __fp16 *)(vectors[2] + i * 2));
                __fp16 h3 = *((const __fp16 *)(vectors[3] + i * 2));
                const float qv = queryPtr[i];
                float d0 = qv - (float)h0; scores[processedCount] += d0 * d0;
                float d1 = qv - (float)h1; scores[processedCount + 1] += d1 * d1;
                float d2 = qv - (float)h2; scores[processedCount + 2] += d2 * d2;
                float d3 = qv - (float)h3; scores[processedCount + 3] += d3 * d3;
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const __fp16*) format.getVectorPointer(internalVectorIds[processedCount]);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int32_t i = 0;
            for (; i <= dim - dimensionBatch; i += dimensionBatch) {
                float32x4_t q0 = vld1q_f32(queryPtr + i);
                float32x4_t q1 = vld1q_f32(queryPtr + i + 4);
                float16x8_t h0 = vld1q_f16((const __fp16 *)(vecPtr + i));
                float32x4_t d0_lo = vcvt_f32_f16(vget_low_f16(h0));
                float32x4_t d0_hi = vcvt_f32_f16(vget_high_f16(h0));
                float32x4_t diff_lo = vsubq_f32(q0, d0_lo);
                float32x4_t diff_hi = vsubq_f32(q1, d0_hi);
                acc = vfmaq_f32(acc, diff_lo, diff_lo);
                acc = vfmaq_f32(acc, diff_hi, diff_hi);
            }

            float finalSum = vaddvq_f32(acc);
            // Scalar tail for dimensions not divisible by 8
            for (; i < dim; ++i) {
                float d = queryPtr[i] - (float)vecPtr[i];
                finalSum += d * d;
            }
            scores[processedCount] = finalSum;
        }

        epilogue.finishInBulk(scores, numVectors);
    }

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        // Lucene's flat FP16 records; the transform is the one this function was instantiated with.
        const LuceneFp16Format format(srchContext);
        const LuceneFp16Format::Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, queryPtr, dim, internalVectorIds, scores, numVectors);
    }
};

//
// SQ (ADC: 4-bit query x 1-bit data) - NEON SIMD implementation
//
// The query is 4-bit quantized and transposed into 4 bit planes (via transposeHalfByte).
// Each bit plane has `binaryCodeBytes` bytes. The int4BitDotProduct computes:
//   Result = popcount(plane0 AND data) * 1
//          + popcount(plane1 AND data) * 2
//          + popcount(plane2 AND data) * 4
//          + popcount(plane3 AND data) * 8
//
// Where a vector's code bytes and correction factors live, and how they combine into a score, is the storage format's
// business (sq_format.h): the kernel below only computes the code dot products.

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

// NEON SIMD batched int4BitDotProduct.
// Uses vcntq_u8 for per-byte popcount on each plane, then weights by 1/2/4/8.
template <int BATCH_SIZE>
static FORCE_INLINE void simd4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t binaryCodeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;

    uint32x4_t acc[BATCH_SIZE];
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        acc[b] = vdupq_n_u32(0);
    }

    int32_t i = 0;
    for ( ; i + 16 <= binaryCodeBytes ; i += 16) {
        // Load 16 bytes from each query plane (shared across all data vectors)
        uint8x16_t q0 = vld1q_u8(plane0 + i);
        uint8x16_t q1 = vld1q_u8(plane1 + i);
        uint8x16_t q2 = vld1q_u8(plane2 + i);
        uint8x16_t q3 = vld1q_u8(plane3 + i);

        // Prefetch next chunk — issued once before the batch loop so we don't
        // flood the prefetch queue (ARM cores typically have 4-8 slots).
        if (i + 16 < binaryCodeBytes) {
            __builtin_prefetch(plane0 + i + 16);
            for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
                __builtin_prefetch(dataVecs[b] + i + 16);
            }
        }

        for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
            // Load 16 bytes of data vector's binary code
            uint8x16_t d = vld1q_u8(dataVecs[b] + i);

            // AND each plane with data, then per-byte popcount
            uint8x16_t p0 = vcntq_u8(vandq_u8(q0, d));
            uint8x16_t p1 = vcntq_u8(vandq_u8(q1, d));
            uint8x16_t p2 = vcntq_u8(vandq_u8(q2, d));
            uint8x16_t p3 = vcntq_u8(vandq_u8(q3, d));

            // Weight: p0*1 + p1*2 + p2*4 + p3*8
            // Max per byte: 8*1 + 8*2 + 8*4 + 8*8 = 120, fits in uint8_t
            uint8x16_t weighted = vaddq_u8(p0, vshlq_n_u8(p1, 1));
            weighted = vaddq_u8(weighted, vshlq_n_u8(p2, 2));
            weighted = vaddq_u8(weighted, vshlq_n_u8(p3, 3));

            // Widen and accumulate: u8 -> u16 -> u32
            acc[b] = vaddq_u32(acc[b], vpaddlq_u16(vpaddlq_u8(weighted)));
        }
    }

    // Horizontal sum into results
    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        results[b] = static_cast<float>(vaddvq_u32(acc[b]));
    }

    // Scalar tail for remaining bytes (< 16)
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

template <SQMetricMode Mode>
struct ArmNeonSQSimilarityFunction final : SimilarityFunction {
    // Scores `internalVectorIds[0..numVectors)` into `scores`. Format (sq_format.h) locates each vector's code bytes
    // and correction factors; Epilogue turns a code dot product and those corrections into a score. Both are
    // compile-time parameters. Mode is not used here; it selects the Lucene epilogue in the two virtuals below.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         const uint8_t* queryPtr,
                                         const int32_t binaryCodeBytes,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_sq_format<Format>::value, "Format must be one of the sq_format.h formats");

        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 8;
        constexpr int32_t vecHalfBlock = 4;
        uint8_t* vectors[vecBlock];

        // Batch size 8
        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecBlock);
            simd4bitDotProductBatch<vecBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                if ((i + 1) < vecBlock) {
                    __builtin_prefetch(format.correctionsAddress(vectors[i + 1], internalVectorIds[processedCount + i + 1]));
                }
                float ax, lx, additional, x1;
                format.readDataCorrections(vectors[i], internalVectorIds[processedCount + i], ax, lx, additional, x1);

                scores[processedCount + i] = epilogue.finish(ax, lx, additional, x1, scores[processedCount + i]);
            }
        }

        // Batch size 4
        for ( ; (processedCount + vecHalfBlock) <= numVectors ; processedCount += vecHalfBlock) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecHalfBlock);
            simd4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
                if ((i + 1) < vecHalfBlock) {
                    __builtin_prefetch(format.correctionsAddress(vectors[i + 1], internalVectorIds[processedCount + i + 1]));
                }
                float ax, lx, additional, x1;
                format.readDataCorrections(vectors[i], internalVectorIds[processedCount + i], ax, lx, additional, x1);

                scores[processedCount + i] = epilogue.finish(ax, lx, additional, x1, scores[processedCount + i]);
            }
        }

        // Tail: remaining vectors (scalar)
        for ( ; processedCount < numVectors ; ++processedCount) {
            const auto* dataVec = format.getVectorPointer(internalVectorIds[processedCount]);
            const float qcDist = static_cast<float>(
                int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

            float ax, lx, additional, x1;
            format.readDataCorrections(dataVec, internalVectorIds[processedCount], ax, lx, additional, x1);

            scores[processedCount] = epilogue.finish(ax, lx, additional, x1, qcDist);
        }
    }

    HOT_SPOT void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                            int32_t* internalVectorIds,
                                            float* scores,
                                            const int32_t numVectors) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t binaryCodeBytes = (srchContext->dimension + 7) / 8;

        // Lucene's interleaved records; the query correction factors come from tmpBuffer.
        const LuceneSqFormat format(srchContext, binaryCodeBytes);
        const LuceneSqFormat::Epilogue<Mode> epilogue(srchContext);

        scoreInBulk(format, epilogue, queryPtr, binaryCodeBytes, internalVectorIds, scores, numVectors);

        if constexpr (Mode == SQMetricMode::MAX_IP) {
            FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk(scores, numVectors);
        } else if constexpr (Mode == SQMetricMode::COSINE) {
            FaissScoreToLuceneScoreTransform::cosineTransformBulk(scores, numVectors);
        } else {
            FaissScoreToLuceneScoreTransform::l2TransformBulk(scores, numVectors);
        }
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) {
        const auto* queryPtr = reinterpret_cast<const uint8_t*>(srchContext->queryVectorSimdAligned);
        const int32_t binaryCodeBytes = (srchContext->dimension + 7) / 8;

        const LuceneSqFormat format(srchContext, binaryCodeBytes);
        const LuceneSqFormat::Epilogue<Mode> epilogue(srchContext);

        const auto* dataVec = format.getVectorPointer(internalVectorId);
        const float qcDist = static_cast<float>(
            int4BitDotProduct(queryPtr, dataVec, binaryCodeBytes));

        float ax, lx, additional, x1;
        format.readDataCorrections(dataVec, internalVectorId, ax, lx, additional, x1);

        const float score = epilogue.finish(ax, lx, additional, x1, qcDist);

        if constexpr (Mode == SQMetricMode::MAX_IP) {
            return FaissScoreToLuceneScoreTransform::ipToMaxIpTransform(score);
        } else if constexpr (Mode == SQMetricMode::COSINE) {
            return FaissScoreToLuceneScoreTransform::cosineTransform(score);
        } else {
            return FaissScoreToLuceneScoreTransform::l2Transform(score);
        }
    }
};


//
// FP16
//
// 1. Max IP
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;
// 3. Cosine (same IP kernel as MaxIP, but with cosine score transform for L2-normalized vectors)
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::cosineTransformBulk, FaissScoreToLuceneScoreTransform::cosineTransform> FP16_COSINE_SIMIL_FUNC;


//
// BF16
// BF16 uses the Faiss SQDistanceComputer for both single and bulk similarity calculations,
// similar to FP16 L2 approach. Faiss internally handles the BF16-to-float conversion.
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonBF16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    // Scores `internalVectorIds[0..numVectors)` into `scores` with the Faiss distance computer. Format (bf16_format.h)
    // locates each vector's BF16 values; Epilogue turns the raw inner products into scores in place. Both are
    // compile-time parameters.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         faiss::ScalarQuantizer::SQDistanceComputer* func,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_bf16_format<Format>::value, "Format must be one of the bf16_format.h formats");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            const auto* vector = format.getVectorPointer(internalVectorIds[i]);
            scores[i] = func->query_to_code(vector);
        }

        epilogue.finishInBulk(scores, numVectors);
    }

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        // Lucene's flat BF16 records; the transform is the one this function was instantiated with.
        const LuceneBf16Format format(srchContext);
        const LuceneBf16Format::Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, func, internalVectorIds, scores, numVectors);
      }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonBF16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    // Scores `internalVectorIds[0..numVectors)` into `scores` with the Faiss distance computer. Format (bf16_format.h)
    // locates each vector's BF16 values; Epilogue turns the raw squared L2 distances into scores in place. Both are
    // compile-time parameters.
    template <typename Format, typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         faiss::ScalarQuantizer::SQDistanceComputer* func,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        static_assert(is_bf16_format<Format>::value, "Format must be one of the bf16_format.h formats");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            const auto* vector = format.getVectorPointer(internalVectorIds[i]);
            scores[i] = func->query_to_code(vector);
        }

        epilogue.finishInBulk(scores, numVectors);
    }

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        // Lucene's flat BF16 records; the transform is the one this function was instantiated with.
        const LuceneBf16Format format(srchContext);
        const LuceneBf16Format::Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, func, internalVectorIds, scores, numVectors);
    }
};

// BF16 similarity function instances
// 1. Max IP
ArmNeonBF16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonBF16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> BF16_L2_SIMIL_FUNC;

//
// SQ
//
// 1. Max IP
ArmNeonSQSimilarityFunction<SQMetricMode::MAX_IP> SQ_IP_SIMIL_FUNC;
// 2. L2
ArmNeonSQSimilarityFunction<SQMetricMode::L2> SQ_L2_SIMIL_FUNC;
// 3. Cosine
ArmNeonSQSimilarityFunction<SQMetricMode::COSINE> SQ_COSINE_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT) {
        return &BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_L2) {
        return &BF16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_IP) {
        return &SQ_IP_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_L2) {
        return &SQ_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_COSINE) {
        return &SQ_COSINE_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_COSINE) {
        return &FP16_COSINE_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
