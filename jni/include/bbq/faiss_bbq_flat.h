#ifndef KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H
#define KNNPLUGIN_JNI_FAISS_BBQ_FLAT_H

#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/MetricType.h"
#include "faiss/impl/DistanceComputer.h"
#include "memory_util.h"

#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace knn_jni {

    template <bool IsMaxIP>
    struct FaissBBQDistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const uint64_t quantizedVectorBytes;
        const uint8_t* data;
        const uint64_t* query;
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
            query = (uint64_t*) x;
            setCorrectionFactors(query, ay, ly, queryAdditional, y1);
        }

        void setCorrectionFactors(const void* target, float& lowerInterval, float& intervalLength, float& additionalCorrection, float& quantizedComponentSum) {
            // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            const auto* correctionFactors = (const float*) ((const uint8_t*) target + quantizedVectorBytes);
            lowerInterval = correctionFactors[0];
            intervalLength = correctionFactors[1] - correctionFactors[0];
            additionalCorrection = correctionFactors[2];
            quantizedComponentSum = *((const int32_t*) (&correctionFactors[3]));
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
            const uint64_t* target = reinterpret_cast<const uint64_t*>(data + i * oneElementByteSize);

            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8

            uint32_t dp = 0;

            for (size_t j = 0; j < words; ++j) {
                dp += __builtin_popcountll(query[j] & target[j]);
            }

            const float score = scoringSecondPart(target, dp);
            return score;
        }

        /// compute distances of current query to 4 stored vectors.
        /// certain DistanceComputer implementations may benefit
        /// heavily from this.
        void distances_batch_4(
                const faiss::idx_t idx0,
                const faiss::idx_t idx1,
                const faiss::idx_t idx2,
                const faiss::idx_t idx3,
                float& dis0,
                float& dis1,
                float& dis2,
                float& dis3) final {
            const uint64_t* target1 = reinterpret_cast<const uint64_t*>(data + idx0 * oneElementByteSize);
            const uint64_t* target2 = reinterpret_cast<const uint64_t*>(data + idx1 * oneElementByteSize);
            const uint64_t* target3 = reinterpret_cast<const uint64_t*>(data + idx2 * oneElementByteSize);
            const uint64_t* target4 = reinterpret_cast<const uint64_t*>(data + idx3 * oneElementByteSize);

            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8

            uint32_t dp1 = 0;
            uint32_t dp2 = 0;
            uint32_t dp3 = 0;
            uint32_t dp4 = 0;

            for (size_t i = 0; i < words; ++i) {
                dp1 += __builtin_popcountll(query[i] & target1[i]);
                dp2 += __builtin_popcountll(query[i] & target2[i]);
                dp3 += __builtin_popcountll(query[i] & target3[i]);
                dp4 += __builtin_popcountll(query[i] & target4[i]);
            }

            dis0 = scoringSecondPart(target1, dp1);
            dis1 = scoringSecondPart(target2, dp2);
            dis2 = scoringSecondPart(target3, dp3);
            dis3 = scoringSecondPart(target4, dp4);
        }

        /// compute distance between two stored vectors
        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
            const uint64_t* target1 = reinterpret_cast<const uint64_t*>(data + i * oneElementByteSize);
            const uint64_t* target2 = reinterpret_cast<const uint64_t*>(data + j * oneElementByteSize);

            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8

            uint32_t dp = 0;

            for (size_t k = 0; k < words; ++k) {
                dp += __builtin_popcountll(target1[k] & target2[k]);
            }

            // Get correction factors
            float ax;
            float lx;
            float additional;
            float x1;
            setCorrectionFactors(target1, ax, lx, additional, x1);

            float az;
            float lz;
            float additionalz;
            float z1;
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
        // For safely casting uint8_t* to float*, we should enforce 4-byte alignment for the vector.
        std::vector<uint8_t, knn_jni::FourBytesAlignedAllocator<uint8_t>> quantizedVectorsAndCorrectionFactors;
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
            // Rewriting code_size, refer below memory layout per each element.
            // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            code_size = _quantizedVectorBytes;
        }

        faiss::DistanceComputer* get_distance_computer() const {
            if (metric_type == faiss::MetricType::METRIC_L2) {
                return new FaissBBQDistanceComputer<false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
            } else if (metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                return new FaissBBQDistanceComputer<true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors);
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
