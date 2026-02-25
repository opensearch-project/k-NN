#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>

#include "parameter_utils.h"
#include "platform_defs.h"
#include "simd/similarity_function/similarity_function.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "jni_util.h"

using knn_jni::simd::similarity_function::SimdVectorSearchContext;
using knn_jni::simd::similarity_function::SimilarityFunction;

// Since Windows CI is failing to recognize std::aligned_malloc, it will make CI pass with macro.
#if defined(_WIN32)
    #include <malloc.h>
#endif

inline void* allocate_aligned_memory(size_t alignment, size_t size) {
#if defined(_WIN32)
    // Works for MSVC and MinGW alike
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void free_aligned_memory(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

//
// SimdVectorSearchContext
//
void SimdVectorSearchContext::getVectorPointersInBulk(uint8_t* vectors[], int32_t* internalVectorIds, int32_t numVectors) {
    if (LIKELY(mmapPages.size() == 1)) {
        // Fast case, there's only one mmap area.
        auto base = mmapPages[0];
        for (int32_t i = 0 ; i < numVectors ; ++i) {
            const uint64_t offset = oneVectorByteSize * internalVectorIds[i];
            if (LIKELY(offset < mmapPageSizes[0])) {
                vectors[i] = reinterpret_cast<uint8_t*>(base) + offset;
            } else {
                throw std::runtime_error(std::string("Offset [") + std::to_string(offset)
                + "] exceeds the chunk size [" + std::to_string(mmapPageSizes[0]) + "].");
            }
        }
        return;
    }

    // There are multiple mapped regions.
    if (mmapPages.empty() == false) {
        for (int32_t i = 0 ; i < numVectors ; ++i) {
            vectors[i] = getVectorPointer(internalVectorIds[i]);
        }
        return;
    }  // End if

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

uint8_t* SimdVectorSearchContext::getVectorPointer(const int32_t internalVectorId) {
    if (LIKELY(mmapPages.size() == 1)) {
        // Fast case, there's only one mmap area.
        return reinterpret_cast<uint8_t*>(mmapPages[0]) + (oneVectorByteSize * internalVectorId);
    }

    if (mmapPages.empty() == false) {
        // Acquire offsets
        const uint64_t startOffset = oneVectorByteSize * internalVectorId;
        const uint64_t endOffsetInclusive = startOffset + oneVectorByteSize - 1;

        // Find region having the vector.
        uint64_t regionStartOffset = 0;
        for (int32_t j = 0 ; j < mmapPageSizes.size() ; ++j) {
            // Note that mmapPageSizes[j] is the endOffset (exclusive) of a region.
            // Therefore, in turn, mmapPageSizes[j - 1] is the starting offset of mmapPageSizes[j] where
            // j > 0, if j == 0, 0 would be the start offset.
            if (startOffset < mmapPageSizes[j]) {
                // Found the first region having the vector.
                // At the worst case, one vector can span across two mapped regions.

                const uint64_t relativeOffsetInFirstRegion = (startOffset - regionStartOffset);

                if (endOffsetInclusive < mmapPageSizes[j]) {
                    // Nice! This region has the entire vector intact.
                    return reinterpret_cast<uint8_t*>(mmapPages[j]) + relativeOffsetInFirstRegion;
                } else {
                    // Prevent seg-fault, this should not happen but it's better to throw an exception than
                    // halting a process.
                    if (UNLIKELY((j + 1) >= mmapPageSizes.size() || (j + 1) >= mmapPages.size())) {
                        throw std::runtime_error(
                        std::string("One vector[vid=") + std::to_string(internalVectorId)
                        + "] straddle two regions(" + std::to_string(j) + "th and " + std::to_string(j + 1)
                        + "th), but there was no next region. We had " + std::to_string(mmapPageSizes.size()) + " regions.");
                    }

                    // No luck, one vector spans across two mapped regions.
                    // We need to copy vectors into a temp buffer having continuous space.
                    // Make sure the vector to have an even address.
                    const int32_t padding = tmpBuffer.size() & 1;
                    const int32_t copyDestIndex = tmpBuffer.size() + padding;
                    tmpBuffer.resize(tmpBuffer.size() + padding + oneVectorByteSize);

                    // Copy the first part
                    const int32_t firstPartSize = mmapPageSizes[j] - startOffset;
                    std::memcpy(&tmpBuffer[copyDestIndex],
                                reinterpret_cast<uint8_t*>(mmapPages[j]) + relativeOffsetInFirstRegion, firstPartSize);

                    // Copy the second part
                    const int32_t secondPartSize = oneVectorByteSize - firstPartSize;
                    if (UNLIKELY(secondPartSize > mmapPageSizes[j + 1])) {
                        throw std::runtime_error(
                            std::string("One vector[vid=") + std::to_string(internalVectorId)
                            + "] straddle two regions(" + std::to_string(j) + "th and " + std::to_string(j + 1)
                            + "th), but the second part of the vector size=" + std::to_string(secondPartSize)
                            + " exceeds the second region size=" + std::to_string(mmapPageSizes[j + 1]));
                    }
                    std::memcpy(&tmpBuffer[copyDestIndex + firstPartSize],
                                reinterpret_cast<uint8_t*>(mmapPages[j + 1]), secondPartSize);

                    // Set the pointer pointing temp buffer.
                    return &tmpBuffer[copyDestIndex];
                }  // End if
            }  // End if

            // mmapPageSizes[j] is the starting offset of mmapPageSizes[j + 1]
            regionStartOffset = mmapPageSizes[j];
        }  // End for

        // Should not happen, region must be found
        std::string errorMsg = std::string("Mapped region for vector(vid=") + std::to_string(internalVectorId) + ") was not found. ";
        errorMsg += "#mmapPageSizes=" + std::to_string(mmapPageSizes.size()) + ", [";
        for (auto pageSize : mmapPageSizes) {
            errorMsg += std::to_string(pageSize) + ", ";
        }
        errorMsg += "], #mmapPages=" + std::to_string(mmapPages.size()) + ", [";
        for (auto pagePtr : mmapPages) {
            errorMsg += std::to_string(reinterpret_cast<uint64_t>(pagePtr)) + ", ";
        }
        errorMsg += "]";
        throw std::runtime_error(std::move(errorMsg));
    }  // End if

    throw std::runtime_error("Search context has not been initialized, mmapPages was empty.");
}

SimdVectorSearchContext::~SimdVectorSearchContext() {
    if (queryVectorSimdAligned) {
        free_aligned_memory(queryVectorSimdAligned);
    }
}

// Thread static local SimdVectorSearchContext
thread_local SimdVectorSearchContext THREAD_LOCAL_SIMD_VEC_SRCH_CTX {};



//
// SimilarityFunction
//
SimdVectorSearchContext* SimilarityFunction::saveSearchContext(
           uint8_t* queryPtr,
           int32_t queryByteSize,
           int32_t dimension,
           int64_t* mmapAddressAndSize,
           int32_t numAddressAndSize,
           int32_t nativeFunctionTypeOrd) {
    // Free tmp buffer
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.tmpBuffer = {};

    // Allocate query vector space
    if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize < queryByteSize) {
        // We need to allocate or re-allocate the space.
        // Allocating 64 bytes aligned memory.
        // Since 16000 dimension is the maximum, therefore at most 62.6KB will be allocated per thread.
        const auto roundedUpQueryByteSize = ((queryByteSize + 63) / 64) * 64;
        void* alignedPtr = allocate_aligned_memory(64, roundedUpQueryByteSize);
        if (alignedPtr == nullptr) {
            throw std::runtime_error(
            std::string("Failed to allocate space for SIMD aligned query vector with size=")
            + std::to_string(queryByteSize));
        }

        // Free up previously allocated space
        if (THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned) {
            free_aligned_memory(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned);
        }

        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned = alignedPtr;
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorByteSize = queryByteSize;
    }

    // Copy query bytes
    std::memcpy(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned, queryPtr, queryByteSize);

    // Set similarity function
    if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT)) {
        // Set similarity function to offload similarity calculation
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction = selectSimilarityFunction(
            NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT);

        // FP16 vector bytes = 2bytes * dimension
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = 2 * dimension;

        // Reset Faiss function for single vector similarity calculation
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction.reset(
             faiss::ScalarQuantizer {static_cast<size_t>(dimension), faiss::ScalarQuantizer::QuantizerType::QT_fp16}
                                    .get_distance_computer(faiss::MetricType::METRIC_INNER_PRODUCT));

        // Assign query to Faiss function
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction->set_query(
            reinterpret_cast<float*>(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned));
    } else if (nativeFunctionTypeOrd == static_cast<int32_t>(NativeSimilarityFunctionType::FP16_L2)) {
        // Set similarity function to offload similarity calculation
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.similarityFunction = selectSimilarityFunction(
            NativeSimilarityFunctionType::FP16_L2);

        // FP16 vector bytes = 2bytes * dimension
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.oneVectorByteSize = 2 * dimension;

        // Reset Faiss function for single vector similarity calculation
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction.reset(
            faiss::ScalarQuantizer {static_cast<size_t>(dimension), faiss::ScalarQuantizer::QuantizerType::QT_fp16}
                                   .get_distance_computer(faiss::MetricType::METRIC_L2));

        // Assign query to Faiss function
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.faissFunction->set_query(
            reinterpret_cast<float*>(THREAD_LOCAL_SIMD_VEC_SRCH_CTX.queryVectorSimdAligned));
    } else {
        throw std::runtime_error(
            std::string("Invalid native similarity function type was given, nativeFunctionTypeOrd=")
            + std::to_string(nativeFunctionTypeOrd));
    }

    // Assign native function ord number
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.nativeFunctionTypeOrd = nativeFunctionTypeOrd;

    // Set dimension
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.dimension = dimension;

    // Set mmap pages
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.clear();
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.clear();
    for (int32_t i = 0 ; i < numAddressAndSize ; i += 2) {
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPages.emplace_back(reinterpret_cast<void*>(mmapAddressAndSize[i]));
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.emplace_back(mmapAddressAndSize[i + 1]);
    }

    // Build prefix sum table. This table will be used to locate the mapped page with a logical offset.
    // For example, let's say the size list was [100, 100, 100] meaning each mmap page had 100 bytes.
    // Then the resulting prefix sum table would be [100, 200, 300]. Then, we can identify the second mmap page has
    // a vector whose start offset is 150.
    for (int32_t i = 1 ; i < THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes.size() ; ++i) {
        THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes[i] += THREAD_LOCAL_SIMD_VEC_SRCH_CTX.mmapPageSizes[i - 1];
    }

    // Return thread_local object
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

SimdVectorSearchContext* SimilarityFunction::getSearchContext() {
    return &THREAD_LOCAL_SIMD_VEC_SRCH_CTX;
}

//
// Similarity function base
//
using BulkScoreTransform = void (*)(float*/*scores*/, int32_t/*num scores to transform*/);
using ScoreTransform = float (*)(float/*score*/);

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct BaseSimilarityFunction : SimilarityFunction {
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
