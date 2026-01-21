#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"
#include "parameter_utils.h"

//
// FP16
//

using BulkScoreTransform = void (*)(float*/*scores*/, int32_t/*num scores to transform*/);
using ScoreTransform = float (*)(float/*score*/);

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct DefaultFP16SimilarityFunction final : SimilarityFunction {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) final {

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

    float calculateSimilarity(SimdVectorSearchContext* srchContext, const int32_t internalVectorId) final {
        // Prepare distance calculation
        auto vector = reinterpret_cast<uint8_t*>(srchContext->getVectorPointer(internalVectorId));
        knn_jni::util::ParameterCheck::require_non_null(vector, "vector from getVectorPointer");
        auto func = dynamic_cast<faiss::ScalarQuantizer::SQDistanceComputer*>(srchContext->faissFunction.get());
        knn_jni::util::ParameterCheck::require_non_null(
            func, "Unexpected distance function acquired. Expected SQDistanceComputer, but it was something else");

        // Calculate distance
        const float score = func->query_to_code(vector);

        // Transform score value if it needs to
        return ScoreTransformFunc(score);
    }
};

//
// FP16
//
// 1. Max IP
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
DefaultFP16SimilarityFunction<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> DEFAULT_FP16_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &DEFAULT_FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &DEFAULT_FP16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif

aaa
