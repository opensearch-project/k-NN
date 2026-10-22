/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_BF16_FORMAT_H
#define OPENSEARCH_KNN_SIMD_BF16_FORMAT_H

#include <cstdint>
#include <type_traits>

#include "platform_defs.h"
#include "simd/similarity_function/similarity_function.h"

namespace knn_jni::simd::similarity_function {

    // Storage formats for the BF16 kernels: the BF16 similarity functions of each ISA file, whose scoreInBulk is
    // templated on the format. Same contract as the FP16 formats (fp16_format.h): a format answers where vector `id`'s
    // bfloat16 values live (getVectorPointersInBulk, getVectorPointer, prefetchAddress) and, through its nested
    // Epilogue, how the raw accumulations become scores (finishInBulk). is_bf16_format (end of file) names the accepted
    // formats for the kernel's static_assert. One format exists today.

    // Lucene's layout: `id` is a slot in the context's vector region(s), 2 * dimension bytes apart, each element the
    // vector's BF16 values back to back.
    struct LuceneBf16Format final {
        SimdVectorSearchContext* ctx;

        FORCE_INLINE explicit LuceneBf16Format(SimdVectorSearchContext* srchContext) : ctx(srchContext) {}

        FORCE_INLINE void getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds,
                                                  const int32_t numVectors) const {
            ctx->getVectorPointersInBulk(vectors, internalVectorIds, numVectors);
        }

        FORCE_INLINE uint8_t* getVectorPointer(const int32_t internalVectorId) const {
            return ctx->getVectorPointer(internalVectorId);
        }

        // Address of vector `id` for a prefetch issued ahead of getVectorPointer(s), or nullptr when the vector straddles
        // two mapped regions (or no region holds it). See LuceneFp16Format::prefetchAddress.
        FORCE_INLINE const uint8_t* prefetchAddress(const int32_t internalVectorId) const {
            return ctx->getVectorPointerIfContiguous(internalVectorId);
        }

        // The Faiss-domain values become Lucene scores in place, with the transform the similarity function was
        // instantiated with (ipToMaxIpTransformBulk or l2TransformBulk).
        template <BulkScoreTransform Transform>
        struct Epilogue final {
            FORCE_INLINE void finishInBulk(float* scores, const int32_t numScores) const {
                Transform(scores, numScores);
            }
        };
    };

    // The formats the BF16 kernels accept (their scoreInBulk static_asserts on it).
    template <class> struct is_bf16_format : std::false_type {};
    template <> struct is_bf16_format<LuceneBf16Format> : std::true_type {};
}

#endif  // OPENSEARCH_KNN_SIMD_BF16_FORMAT_H
