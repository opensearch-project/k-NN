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

// Shared global similarity-function instances and selectSimilarityFunction() for
// the AVX512 / AVX512-SPR builds. Must be included AFTER the including translation
// unit has defined AVX512BF16MaxIP (see avx512_common_simd_similarity_function.cpp).

#ifndef KNN_AVX512_COMMON_REGISTRATION_CPP
#define KNN_AVX512_COMMON_REGISTRATION_CPP

//
// FP16
//
// 1. Max IP
AVX512SPRFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512SPRFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;
// 3. Cosine: uses the native AVX512-FP16 IP kernel when the target has native FP16 (SPR build,
//    __AVX512FP16__), otherwise falls back to the FP32-conversion IP kernel used by FP16 Max IP.
//    Safe because cosine guarantees L2-normalized vectors (||v|| = ||q|| = 1), bounding the dot
//    product to [-1, 1]; the cosine score transform is applied on top of the raw IP result.
#ifdef __AVX512FP16__
AVX512NativeFP16IP<FaissScoreToLuceneScoreTransform::cosineTransformBulk, FaissScoreToLuceneScoreTransform::cosineTransform> FP16_COSINE_SIMIL_FUNC;
#else
AVX512SPRFP16MaxIP<FaissScoreToLuceneScoreTransform::cosineTransformBulk, FaissScoreToLuceneScoreTransform::cosineTransform> FP16_COSINE_SIMIL_FUNC;
#endif

//
// BF16
//
// 1. Max IP
AVX512BF16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512BF16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> BF16_L2_SIMIL_FUNC;

// SQ
//
// 1. Max IP
AVX512SQSimilarityFunction<SQMetricMode::MAX_IP> SQ_IP_SIMIL_FUNC;
// 2. L2
AVX512SQSimilarityFunction<SQMetricMode::L2> SQ_L2_SIMIL_FUNC;
// 3. Cosine
AVX512SQSimilarityFunction<SQMetricMode::COSINE> SQ_COSINE_SIMIL_FUNC;

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

#endif  // KNN_AVX512_COMMON_REGISTRATION_CPP
