#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#endif
#include <arm_neon.h>
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
struct ArmNeonFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {
        int32_t processedCount = 0;
        const __fp16* queryPtr = (const __fp16*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 8;

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const __fp16* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            float16x8_t sum[vecBlock];
            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                sum[v] = vdupq_n_f16(0);
            }

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                float16x8_t q0 = vld1q_f16(queryPtr + i);

                float16x8_t vRegs[vecBlock];
                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    vRegs[v] = vld1q_f16(vectors[v] + i);
                }

                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    sum[v] = vfmaq_f16(sum[v], q0, vRegs[v]);
                }
            }

            // Manual Horizontal Reduction (More template-friendly than vaddvq_f16)
            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                // 1. Fold 8 lanes into 4
                float16x4_t low = vget_low_f16(sum[v]);
                float16x4_t high = vget_high_f16(sum[v]);
                float16x4_t res4 = vadd_f16(low, high);

                // 2. Pairwise add to get scalar
                // vpadd_f16 adds adjacent lanes: [a,b,c,d] -> [a+b, c+d, a+b, c+d]
                float16x4_t res2 = vpadd_f16(res4, res4);
                float16x4_t res1 = vpadd_f16(res2, res2);

                // 3. Extract lane 0
                scores[processedCount + v] = static_cast<float>(vget_lane_f16(res1, 0));
            }
        }

        // Tail loop (simplified for brevity, use same logic as above)
        for (; processedCount < numVectors; ++processedCount) {
            const __fp16* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            float16x8_t acc = vdupq_n_f16(0);
            int32_t i = 0;
            for (; i <= dim - elemPerLoad; i += elemPerLoad) {
                acc = vfmaq_f16(acc, vld1q_f16(queryPtr + i), vld1q_f16(vecPtr + i));
            }
            float16x4_t r4 = vadd_f16(vget_low_f16(acc), vget_high_f16(acc));
            float16x4_t r1 = vpadd_f16(vpadd_f16(r4, r4), vpadd_f16(r4, r4));
            float finalSum = static_cast<float>(vget_lane_f16(r1, 0));
            for (; i < dim; ++i) finalSum += (float)(queryPtr[i] * vecPtr[i]);
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
        int32_t processedCount = 0;
        const __fp16* queryPtr = (const __fp16*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 8;

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const __fp16* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            float16x8_t sum[vecBlock];
            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                sum[v] = vdupq_n_f16(0);
            }

            for (int32_t i = 0; i < dim; i += elemPerLoad) {
                float16x8_t q0 = vld1q_f16(queryPtr + i);

                float16x8_t vRegs[vecBlock];
                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    vRegs[v] = vld1q_f16(vectors[v] + i);
                }

                // Prefetch logic
                if ((i + elemPerLoad) < dim) {
                    #pragma unroll
                    for (int v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + i + elemPerLoad, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                // L2 Math: sum += (q - v) * (q - v)
                #pragma unroll
                for (int v = 0; v < vecBlock; ++v) {
                    float16x8_t diff = vsubq_f16(q0, vRegs[v]);
                    sum[v] = vfmaq_f16(sum[v], diff, diff);
                }
            }

            // Pairwise Horizontal Reduction (Template-safe)
            #pragma unroll
            for (int v = 0; v < vecBlock; ++v) {
                float16x4_t low = vget_low_f16(sum[v]);
                float16x4_t high = vget_high_f16(sum[v]);
                float16x4_t r4 = vadd_f16(low, high);

                // Pairwise fold: [a, b, c, d] -> [a+b, c+d, a+b, c+d]
                float16x4_t r2 = vpadd_f16(r4, r4);
                float16x4_t r1 = vpadd_f16(r2, r2);

                scores[processedCount + v] = static_cast<float>(vget_lane_f16(r1, 0));
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const __fp16* vecPtr = (const __fp16*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            float16x8_t acc = vdupq_n_f16(0);
            int32_t i = 0;

            for (; i <= dim - elemPerLoad; i += elemPerLoad) {
                float16x8_t q = vld1q_f16(queryPtr + i);
                float16x8_t v = vld1q_f16(vecPtr + i);
                float16x8_t diff = vsubq_f16(q, v);
                acc = vfmaq_f16(acc, diff, diff);
            }

            // Scalar reduction for acc
            float16x4_t r4 = vadd_f16(vget_low_f16(acc), vget_high_f16(acc));
            float16x4_t r1 = vpadd_f16(vpadd_f16(r4, r4), vpadd_f16(r4, r4));
            float finalSum = static_cast<float>(vget_lane_f16(r1, 0));

            // Dimension tail
            for (; i < dim; ++i) {
                float d = static_cast<float>(queryPtr[i]) - static_cast<float>(vecPtr[i]);
                finalSum += (d * d);
            }
            scores[processedCount] = finalSum;
        }

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
