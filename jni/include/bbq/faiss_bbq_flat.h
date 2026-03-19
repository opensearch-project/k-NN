#ifndef KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H
#define KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H

#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/MetricType.h"
#include "faiss/impl/DistanceComputer.h"
#include "memory_util.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace knn_jni {

    // Reads correction factors from a potentially unaligned address using std::memcpy.
    // Layout at ptr: [lowerInterval(f32)][upperInterval(f32)][additionalCorrection(f32)][quantizedComponentSum(i32)]
    static inline void readCorrectionFactorsSafe(const uint8_t* ptr, float& lowerInterval,
        float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
        float lower, upper;
        std::memcpy(&lower, ptr, sizeof(float));
        std::memcpy(&upper, ptr + 4, sizeof(float));
        std::memcpy(&additionalCorrection, ptr + 8, sizeof(float));
        int32_t componentSum;
        std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
        lowerInterval = lower;
        intervalLength = upper - lower;
        quantizedComponentSum = static_cast<float>(componentSum);
    }

    // Reads correction factors via direct pointer cast (only safe when ptr is 4-byte aligned).
    static inline void readCorrectionFactorsAligned(const uint8_t* ptr, float& lowerInterval,
        float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
        const auto* correctionFactors = reinterpret_cast<const float*>(ptr);
        lowerInterval = correctionFactors[0];
        intervalLength = correctionFactors[1] - correctionFactors[0];
        additionalCorrection = correctionFactors[2];
        int32_t componentSum;
        std::memcpy(&componentSum, ptr + 12, sizeof(int32_t));
        quantizedComponentSum = static_cast<float>(componentSum);
    }

    template <bool IsMaxIP, bool IsBytesMultipleOf8>
    struct FaissBBQDistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const uint64_t quantizedVectorBytes;
        const uint8_t* data;
        const uint8_t* query;
        const float centroidDp;
        float ay;
        float ly;
        float queryAdditional;
        float y1;
        int32_t dimension;
        int32_t numVectors;

        FaissBBQDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp, int32_t _dimension, int32_t _numVectors)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            // Memory layout : [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            quantizedVectorBytes(_oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t))),
            data((const uint8_t*) _data),
            query(),
            centroidDp(_centroidDp),
            ay(),
            ly(),
            queryAdditional(),
            y1(),
            dimension(_dimension),
            numVectors(_numVectors) {
        }

        void set_query(const float* x) final {
            // The query pointer comes from FaissBBQFlat::quantizedVectorsAndCorrectionFactors
            // which uses NBytesAlignedAllocator<uint8_t, 8> (8-byte aligned base) with a
            // stride of oneElementByteSize that is always a multiple of 8 (quantizedVectorBytes
            // is a multiple of 8 by formula, plus 16 bytes of correction factors).
            // Therefore x is guaranteed 8-byte aligned when IsBytesMultipleOf8 is true.
            query = reinterpret_cast<const uint8_t*>(x);
            setCorrectionFactors(query, ay, ly, queryAdditional, y1);
        }

        void setCorrectionFactors(const void* target, float& lowerInterval, float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
            const uint8_t* ptr = static_cast<const uint8_t*>(target) + quantizedVectorBytes;
            if constexpr (IsBytesMultipleOf8) {
                readCorrectionFactorsAligned(ptr, lowerInterval, intervalLength, additionalCorrection, quantizedComponentSum);
            } else {
                readCorrectionFactorsSafe(ptr, lowerInterval, intervalLength, additionalCorrection, quantizedComponentSum);
            }
        }

        float scoringSecondPart(const void* target, const float dp) {
            // Get correction factors
            float ax;
            float lx;
            float additional;
            float x1;
            setCorrectionFactors(target, ax, lx, additional, x1);

            // Scoring
            float score = ax * ay * dimension
                   + ay * lx * x1
                   + ax * ly * y1
                   + lx * ly * dp;

            if constexpr (IsMaxIP) {
                score += queryAdditional + additional - centroidDp;
            } else {
                // L2: squared distance = ||q||² + ||x||² - 2*dot(q,x)
                // additionalCorrection values carry the squared norms of centroid-centered vectors.
                score = queryAdditional + additional - 2.0f * score;
            }

            return score;
        }

        /// compute distance of vector i to current query
        float operator()(faiss::idx_t i) final {
            const uint8_t* target = data + i * oneElementByteSize;
            // quantizedVectorBytes is always a multiple of 8 per the Java-side contract
            // (FaissService.java: "byte length of a single 1-bit quantized vector, always
            // 64-bit aligned"). The IsBytesMultipleOf8=false path is a safety fallback to
            // avoid UB on unaligned pointer reads, not to handle non-multiple-of-8 sizes.
            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8
            uint32_t dp = 0;

            if constexpr (IsBytesMultipleOf8) {
                const auto* q = reinterpret_cast<const uint64_t*>(query);
                const auto* t = reinterpret_cast<const uint64_t*>(target);
                for (size_t j = 0; j < words; ++j) {
                    dp += __builtin_popcountll(q[j] & t[j]);
                }
            } else {
                // Slower
                for (size_t j = 0; j < words; ++j) {
                    uint64_t queryWord, targetWord;
                    std::memcpy(&queryWord, query + j * 8, sizeof(uint64_t));
                    std::memcpy(&targetWord, target + j * 8, sizeof(uint64_t));
                    dp += __builtin_popcountll(queryWord & targetWord);
                }
            }

            return scoringSecondPart(target, dp);
        }

        /// compute distances of current query to 4 stored vectors.
        void distances_batch_4(
                const faiss::idx_t idx0,
                const faiss::idx_t idx1,
                const faiss::idx_t idx2,
                const faiss::idx_t idx3,
                float& dis0,
                float& dis1,
                float& dis2,
                float& dis3) final {
            const uint8_t* target1 = data + idx0 * oneElementByteSize;
            const uint8_t* target2 = data + idx1 * oneElementByteSize;
            const uint8_t* target3 = data + idx2 * oneElementByteSize;
            const uint8_t* target4 = data + idx3 * oneElementByteSize;

            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8

            uint32_t dp1 = 0, dp2 = 0, dp3 = 0, dp4 = 0;

            if constexpr (IsBytesMultipleOf8) {
                const auto* q = reinterpret_cast<const uint64_t*>(query);
                const auto* t1 = reinterpret_cast<const uint64_t*>(target1);
                const auto* t2 = reinterpret_cast<const uint64_t*>(target2);
                const auto* t3 = reinterpret_cast<const uint64_t*>(target3);
                const auto* t4 = reinterpret_cast<const uint64_t*>(target4);
                for (size_t i = 0; i < words; ++i) {
                    dp1 += __builtin_popcountll(q[i] & t1[i]);
                    dp2 += __builtin_popcountll(q[i] & t2[i]);
                    dp3 += __builtin_popcountll(q[i] & t3[i]);
                    dp4 += __builtin_popcountll(q[i] & t4[i]);
                }
            } else {
                // Slower
                for (size_t i = 0; i < words; ++i) {
                    uint64_t queryWord;
                    std::memcpy(&queryWord, query + i * 8, sizeof(uint64_t));
                    uint64_t w1, w2, w3, w4;
                    std::memcpy(&w1, target1 + i * 8, sizeof(uint64_t));
                    std::memcpy(&w2, target2 + i * 8, sizeof(uint64_t));
                    std::memcpy(&w3, target3 + i * 8, sizeof(uint64_t));
                    std::memcpy(&w4, target4 + i * 8, sizeof(uint64_t));
                    dp1 += __builtin_popcountll(queryWord & w1);
                    dp2 += __builtin_popcountll(queryWord & w2);
                    dp3 += __builtin_popcountll(queryWord & w3);
                    dp4 += __builtin_popcountll(queryWord & w4);
                }
            }

            dis0 = scoringSecondPart(target1, dp1);
            dis1 = scoringSecondPart(target2, dp2);
            dis2 = scoringSecondPart(target3, dp3);
            dis3 = scoringSecondPart(target4, dp4);
        }

        /// compute distance between two stored vectors
        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
            const uint8_t* target1 = data + i * oneElementByteSize;
            const uint8_t* target2 = data + j * oneElementByteSize;

            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8
            uint32_t dp = 0;

            if constexpr (IsBytesMultipleOf8) {
                const auto* t1 = reinterpret_cast<const uint64_t*>(target1);
                const auto* t2 = reinterpret_cast<const uint64_t*>(target2);
                for (size_t k = 0; k < words; ++k) {
                    dp += __builtin_popcountll(t1[k] & t2[k]);
                }
            } else {
                // Slower
                for (size_t k = 0; k < words; ++k) {
                    uint64_t w1, w2;
                    std::memcpy(&w1, target1 + k * 8, sizeof(uint64_t));
                    std::memcpy(&w2, target2 + k * 8, sizeof(uint64_t));
                    dp += __builtin_popcountll(w1 & w2);
                }
            }

            // Get correction factors
            float ax, lx, additional, x1;
            setCorrectionFactors(target1, ax, lx, additional, x1);

            float az, lz, additionalz, z1;
            setCorrectionFactors(target2, az, lz, additionalz, z1);

            // Scoring
            float score = ax * az * dimension
                   + az * lx * x1
                   + ax * lz * z1
                   + lx * lz * dp;

            if constexpr (IsMaxIP) {
                score += additional + additionalz - centroidDp;
            } else {
                score = additional + additionalz - 2 * score;
            }

            return score;
        }
    };

    struct FaissBBQFlat final : faiss::IndexBinary {
        int64_t numVectors;
        int32_t quantizedVectorBytes;
        float centroidDp;
        int32_t oneElementSize;
        // For safely casting uint8_t* to float*, we should enforce 8-byte alignment for the vector.
        std::vector<uint8_t, knn_jni::NBytesAlignedAllocator<uint8_t, 8>> quantizedVectorsAndCorrectionFactors;
        int32_t dimension;

        FaissBBQFlat(int64_t _numVectors, int32_t _quantizedVectorBytes, float _centroidDp, int32_t _dimension, faiss::MetricType _metric)
          : faiss::IndexBinary(_dimension, _metric),
            numVectors(_numVectors),
            quantizedVectorBytes(_quantizedVectorBytes),
            centroidDp(_centroidDp),
            oneElementSize(_quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t)),
            // Pre allocate vector storage space
            quantizedVectorsAndCorrectionFactors(_numVectors * oneElementSize),
            dimension(_dimension) {

            // Just changing the size, not shrinking.
            quantizedVectorsAndCorrectionFactors.resize(0);
            // Rewriting code_size to the full element size so that hnsw_add_vertices
            // strides correctly through the packed buffer when computing:
            //   x + (pt_id - n0) * index_hnsw.code_size
            // Memory layout per element:
            // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            code_size = oneElementSize;
        }

        faiss::DistanceComputer* get_distance_computer() const {
            const bool aligned = (oneElementSize % 8) == 0;
            if (metric_type == faiss::MetricType::METRIC_L2) {
                if (aligned) {
                    return new FaissBBQDistanceComputer<false, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
                } else {
                    return new FaissBBQDistanceComputer<false, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
                }
            } else if (metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                if (aligned) {
                    return new FaissBBQDistanceComputer<true, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
                } else {
                    return new FaissBBQDistanceComputer<true, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
                }
            }

            throw std::runtime_error("Unsupported metric type - " + std::to_string(metric_type));
        }

        void search(faiss::idx_t n,
                    const uint8_t* x,
                    faiss::idx_t k,
                    int32_t* distances,
                    faiss::idx_t* labels,
                    const faiss::SearchParameters* params = nullptr) const final {
            throw std::runtime_error("FaissBBQFlat does not support search");
        }

        void reset() final {
            throw std::runtime_error("FaissBBQFlat does not support reset");
        }

        void merge_from(faiss::IndexBinary& otherIndex, faiss::idx_t add_id = 0) final {
            throw std::runtime_error("FaissBBQFlat does not support merge_from");
        };

        void add(faiss::idx_t n, const uint8_t* x) final {
            // We only increase ntotal here, as it does not actually own the binary quantized vectors.
            // They are in a separate files.
            ntotal += n;
        }
    };

}

#endif //KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H
