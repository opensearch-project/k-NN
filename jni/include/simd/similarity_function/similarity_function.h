#ifndef OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
#define OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H

#include <cstdint>
#include <vector>
#include <memory>
#include "faiss/impl/DistanceComputer.h"

namespace knn_jni::simd::similarity_function {
    enum class NativeSimilarityFunctionType {
        // Max inner product for FP16.
        // Max inner product will transform inner product to v < 0 ? 1 / (1 - v) : (1 + v)
        FP16_MAXIMUM_INNER_PRODUCT,
        // L2 for FP16
        FP16_L2
    };

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
        // Temp buffer which is reset per search.
        std::vector<uint8_t> tmpBuffer;

        ~SimdVectorSearchContext();

        // This will look up internal mapping table and acquire raw pointers pointing to vectors with the passed vector
        // ids then put them into `vectors`
        void getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds, int32_t numVectors);

        // Similar to `getVectorPointersInBulk`, but it returns raw pointer pointing to the vector it's looking for.
        uint8_t* getVectorPointer(int32_t internalVectorId);
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
