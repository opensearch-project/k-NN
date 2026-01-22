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

inline void print_f32x4(const char* label, float32x4_t v) {
    alignas(16) float tmp[4];
    vst1q_f32(tmp, v);

    std::cout << label << ": [ "
              << tmp[0] << ", "
              << tmp[1] << ", "
              << tmp[2] << ", "
              << tmp[3] << " ]\n";
}


//
// FP16
//

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        int32_t processedCount = 0;
        const float* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 4;

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const __fp16* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            float32x4_t sum[vecBlock];
            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                sum[v] = vdupq_n_f32(0.0f);
            }

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                // 1. LOAD & CONVERT Phase
                // Load 4 FP32 from query
                float32x4_t q0 = vld1q_f32(queryPtr + i);

                float32x4_t vRegs32[vecBlock];
                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    // Load 4 FP16 and convert to FP32 immediately
                    vRegs32[v] = vcvt_f32_f16(vld1_f16(vectors[v] + i));
                }

                // 2. PREFETCH Phase
                // We prefetch the NEXT chunk of data.
                // Query is float (4 bytes): 4 elements = 16 bytes.
                // Vectors are fp16 (2 bytes): 4 elements = 8 bytes.
                if ((i + elemPerLoad) < dim) {
                    #pragma unroll
                    for (int v = 0; v < vecBlock; ++v) {
                        // Prefetching 32 bytes ahead is usually optimal for Neon L1
                        __builtin_prefetch(vectors[v] + i + (elemPerLoad * 4), 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + (elemPerLoad * 4), 0, 3);
                }

                // 3. FMA Phase
                // While the prefetch is issued and conversions are finishing in the pipeline...
                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    sum[v] = vfmaq_f32(sum[v], q0, vRegs32[v]);
                }
            }

            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = vaddvq_f32(sum[v]);
            }
        }
        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const __fp16* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            float32x4_t acc = vdupq_n_f32(0.0f);
            int32_t i = 0;
            for (; i <= dim - elemPerLoad; i += elemPerLoad) {
                float32x4_t q = vld1q_f32(queryPtr + i);
                float32x4_t v = vcvt_f32_f16(vld1_f16(vecPtr + i));
                acc = vfmaq_f32(acc, q, v);
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
// FP16
//
// 1. Max IP
ArmNeonFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
ArmNeonFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;

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
