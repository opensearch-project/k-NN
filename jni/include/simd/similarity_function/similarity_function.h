#ifndef OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
#define OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H

#include <cstdint>
#include <vector>
#include <memory>
#include "faiss/impl/DistanceComputer.h"

namespace knn_jni::simd::similarity_function {
    enum class NativeSimilarityFunctionType {
        FP16_MAXIMUM_INNER_PRODUCT,
        FP16_L2
    };

    struct SimilarityFunction;

    struct SimdVectorSearchContext {
        void* queryVectorSimdAligned = nullptr;
        int32_t queryVectorByteSize = 0;
        int32_t dimension = 0;
        int64_t oneVectorByteSize = 0;
        std::vector<void*> mmapPages;
        std::vector<int64_t> mmapPageSizes;
        int32_t nativeFunctionTypeOrd = -1;
        SimilarityFunction* similarityFunction;
        std::unique_ptr<faiss::DistanceComputer> faissFunction;
        std::vector<uint8_t> tmpBuffer;

        ~SimdVectorSearchContext();

        void getVectorPointersInBulk(uint8_t* vector[], int32_t* internalVectorIds, int32_t numVectors);

        uint8_t* getVectorPointer(int32_t internalVectorId);
    };

    struct SimilarityFunction {
        virtual ~SimilarityFunction() = default;

        static SimdVectorSearchContext* saveSearchContext(
                   uint8_t* queryPtr,
                   int32_t queryByteSize,
                   int32_t dimension,
                   int64_t* mmapAddressAndSize,
                   int32_t numAddressAndSize,
                   int32_t nativeFunctionTypeOrd);

        static SimdVectorSearchContext* getSearchContext();

        virtual void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                               int32_t* internalVectorIds,
                                               float* scores,
                                               int32_t numVectors) = 0;

        virtual float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) = 0;

      private:
        static SimilarityFunction* selectSimilarityFunction(NativeSimilarityFunctionType nativeFunctionType);
    };
}

#endif  // OPENSEARCH_KNN_SIMD_SIMILARITY_FUNCTION_H
