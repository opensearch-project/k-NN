/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
#define OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H

#include <cstdint>
#include <vector>
#include <memory>
#include "faiss/impl/DistanceComputer.h"
#include "memory_util.h"
#include "platform_defs.h"

namespace knn_jni::simd::similarity_function {
    // The integer a Java caller passes as `nativeFunctionTypeOrd` is part of the JNI wire contract. Each constant's value
    // mirrors SimdVectorComputeService.SimilarityFunctionType#getWireValue() on the Java side: never renumber an
    // existing constant, and give any new constant the next unused number regardless of where it is declared.
    enum class NativeSimilarityFunctionType : int32_t {
        // Max inner product for FP16.
        // Max inner product will transform inner product to v < 0 ? 1 / (1 - v) : (1 + v)
        FP16_MAXIMUM_INNER_PRODUCT = 0,
        // L2 for FP16
        FP16_L2 = 1,
        SQ_IP = 2,
        SQ_L2 = 3,
        // Max inner product for BF16.
        BF16_MAXIMUM_INNER_PRODUCT = 4,
        // L2 for BF16
        BF16_L2 = 5,
        // Cosine for FP16. Vectors are L2-normalized so cosine = inner product with score = (1 + dot) / 2.
        FP16_COSINE = 6,
        // Cosine for SQ. Same IP intermediate math as SQ_IP but with cosine score transform.
        SQ_COSINE = 7
    };

    // Metric mode for SQ (Scalar Quantized) similarity functions.
    // Controls both intermediate score computation and final score transformation.
    enum class SQMetricMode {
        MAX_IP,   // Inner product: score += correction; ipToMaxIpTransform
        L2,       // L2 distance: score = max(0, correction - 2*score); l2Transform
        COSINE    // Cosine (normalized IP): score += correction; cosineTransform
    };

    // Score transforms a similarity function is instantiated with (see faiss_score_to_lucene_transform.cpp). The bulk
    // form rewrites `scores[0..numScores)` in place; the single form maps one value.
    using BulkScoreTransform = void (*)(float* /*scores*/, int32_t /*num scores to transform*/);
    using ScoreTransform = float (*)(float /*score*/);

    struct SimilarityFunction;

    struct SimdVectorSearchContext {
        // SIMD aligned query bytes
        void* queryVectorSimdAligned = nullptr;
        // Query vector byte size
        int32_t queryVectorByteSize = 0;
        // Vector dimension
        int32_t dimension = 0;
        // Stored vector byte size. Based on its quantization status, this value can vary than `queryVectorByteSize`.
        // For example, for FP16, this value would be 2 * dimension
        int64_t oneVectorByteSize = 0;
        // Underlying mmap page table. If Faiss index is large, then there can be several mapped regions over it.
        std::vector<void*> mmapPages;
        // Mapped page size for each. mmapPageSizes[i] -> mmapPages[i]'s size
        std::vector<int64_t> mmapPageSizes;
        // Function type index
        int32_t nativeFunctionTypeOrd = -1;
        // Similarity function calculating similarity that was chosen based on `nativeFunctionTypeOrd`.
        SimilarityFunction* similarityFunction;
        // Faiss distance computation function.
        std::unique_ptr<faiss::DistanceComputer> faissFunction;
        // Temp buffer which is reset per search. Also used by getVectorPointer() to reassemble a
        // vector that straddles two mmap regions, so it may be resized (and reallocated) mid-scan.
        std::vector<uint8_t> tmpBuffer;
        // Dedicated, page-aligned buffer holding the format-converted query (FP16 or BF16) for the
        // native SIMD kernels. A single buffer suffices because a given search uses exactly one
        // quantization format, and it is reused across bulk-scoring calls so the query is not
        // reallocated on every invocation. Kept separate from tmpBuffer so that a cross-region
        // vector reassembly (which resizes tmpBuffer) can never invalidate the query pointer held
        // by the kernel.
        std::vector<uint8_t, knn_jni::NBytesAlignedAllocator<uint8_t, 4096>> queryConvertedBuffer;

        ~SimdVectorSearchContext();

        // This will look up internal mapping table and acquire raw pointers pointing to vectors with the passed vector
        // ids then put them into `vectors`
        void getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds, int32_t numVectors);

        // Similar to `getVectorPointersInBulk`, but it returns raw pointer pointing to the vector it's looking for.
        uint8_t* getVectorPointer(int32_t internalVectorId);

        // Address of vector `internalVectorId` when it lies entirely inside one mapped region, nullptr when it straddles
        // two regions or no region holds it. getVectorPointer reassembles a straddling vector into tmpBuffer, which may
        // reallocate; this method never touches tmpBuffer, so it is safe to call while pointers handed out by
        // getVectorPointer(s) are live, e.g. to prefetch vectors that will be scored shortly.
        FORCE_INLINE const uint8_t* getVectorPointerIfContiguous(const int32_t internalVectorId) const {
            const uint64_t startOffset = static_cast<uint64_t>(oneVectorByteSize) * internalVectorId;
            if (LIKELY(mmapPages.size() == 1)) {
                // Fast case, there's only one mmap area: the same address getVectorPointer returns.
                return reinterpret_cast<const uint8_t*>(mmapPages[0]) + startOffset;
            }

            // mmapPageSizes[j] is the exclusive end offset of region j (saveSearchContext turns the sizes into a prefix
            // sum), so the region holding startOffset is the first one whose end lies beyond it. The vector is addressable
            // only when it also ends inside that region.
            const uint64_t endOffsetInclusive = startOffset + oneVectorByteSize - 1;
            uint64_t regionStartOffset = 0;
            for (size_t j = 0 ; j < mmapPageSizes.size() ; ++j) {
                const auto regionEndOffset = static_cast<uint64_t>(mmapPageSizes[j]);
                if (startOffset < regionEndOffset) {
                    if (endOffsetInclusive < regionEndOffset) {
                        return reinterpret_cast<const uint8_t*>(mmapPages[j]) + (startOffset - regionStartOffset);
                    }
                    // Straddles regions j and j + 1.
                    return nullptr;
                }
                regionStartOffset = regionEndOffset;
            }
            return nullptr;
        }
    };

    // This class's responsibility is to calculate similarity between query and vectors.
    // It first saves search context first in static thread local object, then use it during search.
    struct SimilarityFunction {
        virtual ~SimilarityFunction() = default;

        // Save required information during search in static thread local storage.
        // The maximum thread local storage is bounded by O(SizeOf(Query Vector Size)).
        static SimdVectorSearchContext* saveSearchContext(
                   uint8_t* queryPtr,
                   int32_t queryByteSize,
                   int32_t dimension,
                   int64_t* mmapAddressAndSize,
                   int32_t numAddressAndSize,
                   int32_t nativeFunctionTypeOrd);

        // Return thread static storage it's holding.
        static SimdVectorSearchContext* getSearchContext();

        // Repoints the already-saved search context at a new vector chunk
        static void updateVectorChunk(uint8_t* vectorsPtr, int64_t vectorsByteSize);

        // Given vector ids, calculate similarity in bulk and put scores into `scores`.
        virtual void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                               int32_t* internalVectorIds,
                                               float* scores,
                                               int32_t numVectors) = 0;

        // Similar to `calculateSimilarityInBulk`, but this targets a single vector and returns a score.
        virtual float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) = 0;

      private:
        // Select similarity function based on the function type.
        static SimilarityFunction* selectSimilarityFunction(NativeSimilarityFunctionType nativeFunctionType);
    };
}

#endif  // OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
