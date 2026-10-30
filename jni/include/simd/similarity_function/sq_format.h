/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_SQ_FORMAT_H
#define OPENSEARCH_KNN_SIMD_SQ_FORMAT_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "platform_defs.h"
#include "simd/similarity_function/similarity_function.h"

namespace knn_jni::simd::similarity_function {

    // Storage formats for the 1-bit scalar-quantized (ADC) kernel: the SQ similarity function of each ISA file, whose
    // scoreInBulk is templated on the format. A format answers where vector `id`'s code bytes and correction factors
    // live (getVectorPointersInBulk, getVectorPointer, readDataCorrections, correctionsAddress) and, through its nested
    // Epilogue, how a code dot product and those corrections become a score (finish). The kernel never assumes a layout
    // itself, so a format that stores the same information differently (say, corrections in separate columnar arrays
    // rather than inline after each code) plugs in without touching the intrinsics. No common base, no virtual
    // function: the kernel is instantiated per format, and every member is FORCE_INLINE so the format folds into the
    // loop. is_sq_format (end of file) names the accepted formats for the kernel's static_assert.

    // The query is 4-bit quantized and transposed into 4 bit planes; the code dot product weights them 1, 2, 4, 8.
    constexpr float FOUR_BIT_SCALE = 1.0f / 15.0f;

    // Lucene's layout: `id` is a slot in the context's vector region(s), oneVectorByteSize apart. Each element is
    // [code (binaryCodeBytes) | lowerInterval f32 | upperInterval f32 | additionalCorrection f32 | quantizedComponentSum i32].
    // Because oneVectorByteSize need not be a multiple of 4, elements can start at non-4-byte-aligned offsets, hence
    // memcpy for the corrections rather than a reinterpret_cast<float*>.
    struct LuceneSqFormat final {
        SimdVectorSearchContext* ctx;
        int32_t binaryCodeBytes;

        FORCE_INLINE LuceneSqFormat(SimdVectorSearchContext* srchContext, const int32_t codeBytes)
            : ctx(srchContext), binaryCodeBytes(codeBytes) {}

        FORCE_INLINE void getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds,
                                                  const int32_t numVectors) const {
            ctx->getVectorPointersInBulk(vectors, internalVectorIds, numVectors);
        }

        FORCE_INLINE uint8_t* getVectorPointer(const int32_t internalVectorId) const {
            return ctx->getVectorPointer(internalVectorId);
        }

        // Reads the correction factors that follow the code inside the same element. `ax` is the lower interval, `lx`
        // is upper - lower, `additional` the additional correction and `x1` the quantized component sum as a float.
        FORCE_INLINE void readDataCorrections(const uint8_t* dataVec, const int32_t /*internalVectorId*/,
                                              float& ax, float& lx, float& additional, float& x1) const {
            const uint8_t* ptr = dataVec + binaryCodeBytes;
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

        // Address to prefetch before readDataCorrections of the same vector.
        FORCE_INLINE const void* correctionsAddress(const uint8_t* dataVec, const int32_t /*internalVectorId*/) const {
            return dataVec + binaryCodeBytes;
        }

        // The query correction factors from the context's tmpBuffer (written by saveSQSearchContext) and the ADC score
        // math. `finish` returns the Faiss-domain value; the caller applies the FaissScoreToLuceneScoreTransform that
        // matches Mode.
        template <SQMetricMode Mode>
        struct Epilogue final {
            const int32_t dim;
            float ay;
            float ly;
            float queryAdditional;
            float y1;
            float centroidDp;

            FORCE_INLINE explicit Epilogue(const SimdVectorSearchContext* srchContext) : dim(srchContext->dimension) {
                const auto* queryCorrectionPtr = reinterpret_cast<const float*>(srchContext->tmpBuffer.data());
                ay = queryCorrectionPtr[0];
                ly = (queryCorrectionPtr[1] - queryCorrectionPtr[0]) * FOUR_BIT_SCALE;
                queryAdditional = queryCorrectionPtr[2];
                int32_t y1Raw; std::memcpy(&y1Raw, &queryCorrectionPtr[3], sizeof(int32_t));
                y1 = static_cast<float>(y1Raw);
                centroidDp = queryCorrectionPtr[4];
            }

            // `lx` is upper - lower, `x1` the component sum and `qcDist` the code dot product, each as a float.
            FORCE_INLINE float finish(const float ax, const float lx, const float additional, const float x1,
                                      const float qcDist) const {
                float score = ax * ay * dim
                            + ay * lx * x1
                            + ax * ly * y1
                            + lx * ly * qcDist;

                if constexpr (Mode == SQMetricMode::MAX_IP || Mode == SQMetricMode::COSINE) {
                    score += queryAdditional + additional - centroidDp;
                } else {
                    score = std::max(0.0F, queryAdditional + additional - 2 * score);
                }
                return score;
            }
        };
    };

    // The formats the SQ kernel accepts (its scoreInBulk static_asserts on it).
    template <class> struct is_sq_format : std::false_type {};
    template <> struct is_sq_format<LuceneSqFormat> : std::true_type {};
}

#endif  // OPENSEARCH_KNN_SIMD_SQ_FORMAT_H
