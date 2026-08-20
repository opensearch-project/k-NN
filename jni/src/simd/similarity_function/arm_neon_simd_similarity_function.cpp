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



//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Bulk inner product with 4 batch
        int32_t processedCount = 0;
        constexpr int32_t vecBlock = 4;
        const uint8_t* vectors[vecBlock];
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;
        constexpr int32_t dimensionBatch = 8;

        for ( ; (processedCount + vecBlock) <= numVectors ; processedCount += vecBlock) {
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

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
            const auto* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
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

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            // Calculate distance
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->query_to_code(vector);
        }

        // Transform score values if it needs to
        BulkScoreTransformFunc(scores, numVectors);
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

template <bool IsMaxIP>
struct ArmNeonSQSimilarityFunction final : SimilarityFunction {
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
            simd4bitDotProductBatch<vecBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

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
            simd4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

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
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;


//
// BF16
// BF16 uses the Faiss SQDistanceComputer for both single and bulk similarity calculations,
// similar to FP16 L2 approach. Faiss internally handles the BF16-to-float conversion.
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonBF16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->query_to_code(vector);
        }

        BulkScoreTransformFunc(scores, numVectors);
      }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonBF16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        // Prepare similarity calculation
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        for (int32_t i = 0 ; i < numVectors ; ++i) {
            auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorIds[i]));
            scores[i] = func->query_to_code(vector);
        }

        BulkScoreTransformFunc(scores, numVectors);
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
ArmNeonSQSimilarityFunction<true> SQ_IP_SIMIL_FUNC;
// 2. L2
ArmNeonSQSimilarityFunction<false> SQ_L2_SIMIL_FUNC;

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
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
