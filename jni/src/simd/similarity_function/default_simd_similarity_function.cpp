/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"
#include "parameter_utils.h"
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
// Default (non-SIMD) similarity function backed by Faiss's SQDistanceComputer.
// Generic over the storage format (fp16_format.h / bf16_format.h, used for both FP16 and BF16), since the format
// only has to say where a vector lives: the actual decoding is delegated to the Faiss distance computer.
//
template <typename Format, BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct DefaultFaissSimilarityFunction final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    static_assert(is_fp16_format<Format>::value || is_bf16_format<Format>::value,
                  "Format must be one of the fp16_format.h or bf16_format.h formats");

    // Scores `internalVectorIds[0..numVectors)` into `scores` with the Faiss distance computer. Format locates each
    // vector's values; Epilogue turns the raw distances into scores in place. Both are compile-time parameters.
    template <typename Epilogue>
    static FORCE_INLINE void scoreInBulk(const Format& format,
                                         const Epilogue& epilogue,
                                         faiss::ScalarQuantizer::SQDistanceComputer* func,
                                         int32_t* internalVectorIds,
                                         float* scores,
                                         const int32_t numVectors) {
        for (int32_t i = 0 ; i < numVectors ; ++i) {
            // Calculate distance
            const auto* vector = format.getVectorPointer(internalVectorIds[i]);
            scores[i] = func->query_to_code(vector);
        }

        // Transform score values if it needs to
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

        // Lucene's flat records; the transform is the one this function was instantiated with.
        const Format format(srchContext);
        const typename Format::template Epilogue<BulkScoreTransformFunc> epilogue{};

        scoreInBulk(format, epilogue, func, internalVectorIds, scores, numVectors);
    }
};

//
// FP16
//
// 1. Max IP
DefaultFaissSimilarityFunction<LuceneFp16Format, FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
DefaultFaissSimilarityFunction<LuceneFp16Format, FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> DEFAULT_FP16_L2_SIMIL_FUNC;
// 3. Cosine (same IP kernel, but with cosine score transform for L2-normalized vectors)
DefaultFaissSimilarityFunction<LuceneFp16Format, FaissScoreToLuceneScoreTransform::cosineTransformBulk, FaissScoreToLuceneScoreTransform::cosineTransform> DEFAULT_FP16_COSINE_SIMIL_FUNC;

//
// BF16 - Uses the same default (Faiss SQDistanceComputer-based) approach as FP16
//
// 1. Max IP
DefaultFaissSimilarityFunction<LuceneBf16Format, FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> DEFAULT_BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
DefaultFaissSimilarityFunction<LuceneBf16Format, FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> DEFAULT_BF16_L2_SIMIL_FUNC;

//
// SQ (ADC: 4-bit query x 1-bit data) - Default (non-SIMD) implementation
//
// Pure C++ implementation written for compiler auto-vectorization with -O3.
// No SIMD intrinsics. Uses __builtin_popcount[ll] and std::memcpy.
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

// Scalar int4BitDotProduct
// q has 4 * binaryCodeBytes bytes (4 bit planes), d has binaryCodeBytes bytes.
// Uses std::memcpy for uint64_t loads to avoid undefined behavior from unaligned
// reinterpret_cast when binaryCodeBytes is not a multiple of 8.
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

// Default (non-SIMD) batched int4BitDotProduct.
// Processes one batch element at a time with the byte loop as the inner loop,
// giving the compiler the best auto-vectorization opportunity across the byte dimension.
template <int BATCH_SIZE>
static FORCE_INLINE void default4bitDotProductBatch(
    const uint8_t* queryPtr,
    uint8_t** dataVecs,
    const int32_t binaryCodeBytes,
    float* results) {

    const uint8_t* plane0 = queryPtr;
    const uint8_t* plane1 = queryPtr + binaryCodeBytes;
    const uint8_t* plane2 = queryPtr + 2 * binaryCodeBytes;
    const uint8_t* plane3 = queryPtr + 3 * binaryCodeBytes;

    const int32_t words = binaryCodeBytes >> 3;
    const int32_t remainStart = words * 8;

    for (int32_t b = 0 ; b < BATCH_SIZE ; ++b) {
        int64_t acc = 0;
        const uint8_t* data = dataVecs[b];

        // 8-byte word loop — compiler can auto-vectorize this
        for (int32_t w = 0 ; w < words ; ++w) {
            const int32_t offset = w * 8;
            uint64_t dWord;
            std::memcpy(&dWord, data + offset, sizeof(uint64_t));

            uint64_t q0, q1, q2, q3;
            std::memcpy(&q0, plane0 + offset, sizeof(uint64_t));
            std::memcpy(&q1, plane1 + offset, sizeof(uint64_t));
            std::memcpy(&q2, plane2 + offset, sizeof(uint64_t));
            std::memcpy(&q3, plane3 + offset, sizeof(uint64_t));

            acc += __builtin_popcountll(q0 & dWord) * 1
                 + __builtin_popcountll(q1 & dWord) * 2
                 + __builtin_popcountll(q2 & dWord) * 4
                 + __builtin_popcountll(q3 & dWord) * 8;
        }

        // Byte remainder — handles non-multiple-of-8 binaryCodeBytes
        for (int32_t r = remainStart ; r < binaryCodeBytes ; ++r) {
            uint8_t db = data[r];
            acc += __builtin_popcount((plane0[r] & db) & 0xFF) * 1
                 + __builtin_popcount((plane1[r] & db) & 0xFF) * 2
                 + __builtin_popcount((plane2[r] & db) & 0xFF) * 4
                 + __builtin_popcount((plane3[r] & db) & 0xFF) * 8;
        }

        results[b] = static_cast<float>(acc);
    }
}

template <SQMetricMode Mode>
struct DefaultSQSimilarityFunction final : SimilarityFunction {
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
            default4bitDotProductBatch<vecBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            for (int32_t i = 0 ; i < vecBlock ; ++i) {
                float ax, lx, additional, x1;
                format.readDataCorrections(vectors[i], internalVectorIds[processedCount + i], ax, lx, additional, x1);

                scores[processedCount + i] = epilogue.finish(ax, lx, additional, x1, scores[processedCount + i]);
            }
        }

        // Batch size 4
        for ( ; (processedCount + vecHalfBlock) <= numVectors ; processedCount += vecHalfBlock) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[processedCount], vecHalfBlock);
            default4bitDotProductBatch<vecHalfBlock>(queryPtr, vectors, binaryCodeBytes, &scores[processedCount]);

            for (int32_t i = 0 ; i < vecHalfBlock ; ++i) {
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
// SQ
//
// 1. Max IP
DefaultSQSimilarityFunction<SQMetricMode::MAX_IP> SQ_IP_SIMIL_FUNC;
// 2. L2
DefaultSQSimilarityFunction<SQMetricMode::L2> SQ_L2_SIMIL_FUNC;
// 3. Cosine
DefaultSQSimilarityFunction<SQMetricMode::COSINE> SQ_COSINE_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &DEFAULT_FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT) {
        return &DEFAULT_BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_L2) {
        return &DEFAULT_BF16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_IP) {
        return &SQ_IP_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_L2) {
        return &SQ_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::SQ_COSINE) {
        return &SQ_COSINE_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_COSINE) {
        return &DEFAULT_FP16_COSINE_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
