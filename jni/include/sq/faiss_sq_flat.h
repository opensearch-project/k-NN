#ifndef KNNPLUGIN_JNI_FAISS_SQ_FLAT_H
#define KNNPLUGIN_JNI_FAISS_SQ_FLAT_H

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

    // Only bit widths supported by the memory-optimized-search (MOS) path.
    // Kept in sync with FaissSQEncoder.isMosBits on the Java side and with
    // ScalarEncodingResolver.SUPPORTED_DOC_BITS.
    static inline bool isMosDocBits(int32_t bits) {
        return bits == 1 || bits == 2 || bits == 4;
    }

    // Validates a docBits value and returns it unchanged when supported. Called from
    // FaissSQDistanceComputer's initializer list so the throw beats the `qvb / docBits`
    // computation in `planeBytes`, which would otherwise trigger a SIGFPE for docBits == 0.
    static inline int32_t validateDocBits(int32_t bits) {
        if (!isMosDocBits(bits)) {
            throw std::runtime_error(
                "FaissSQDistanceComputer: unsupported docBits=" + std::to_string(bits)
                + ". Supported: 1, 2, 4."
            );
        }
        return bits;
    }

    // Validates that quantizedVectorBytes is compatible with the given docBits and returns it
    // unchanged. For docBits == 2 the bit-plane kernel requires equal-length planes (qvb must
    // be even). For docBits ∈ {1, 4} any positive qvb is valid — B=1 is a single plane, and
    // B=4 uses PACKED_NIBBLE where each byte independently carries two elements.
    static inline uint64_t validateQvbForDocBits(uint64_t qvb, int32_t bits) {
        if (bits == 2 && qvb % 2 != 0) {
            throw std::runtime_error(
                "FaissSQDistanceComputer: quantizedVectorBytes=" + std::to_string(qvb)
                + " must be even for docBits=2 (two bit planes of equal length)"
            );
        }
        return qvb;
    }

    template <bool IsMaxIP, bool IsBytesMultipleOf8>
    struct FaissSQDistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const uint64_t quantizedVectorBytes;
        // Number of bits used to quantize each document dimension (1, 2, or 4).
        const int32_t docBits;
        // Byte length of a single bit plane. Only meaningful for the bit-plane popcount path
        // (B=1, B=2): the quantized code is docBits contiguous planes, each planeBytes long,
        // so quantizedVectorBytes == docBits * planeBytes. Unused for B=4 (PACKED_NIBBLE),
        // where `bothPackedNibbleDp` iterates over quantizedVectorBytes directly.
        const uint64_t planeBytes;
        // Reconstruction scale for the quantization interval: 1 / (2^docBits - 1). For docBits == 1
        // this is 1, preserving the legacy single-bit behavior.
        const float dataScale;
        const uint8_t* data;
        const uint8_t* query;
        const float centroidDp;
        float ay;
        float ly;
        float queryAdditional;
        float y1;
        int32_t dimension;
        int32_t numVectors;

        FaissSQDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp, int32_t _dimension, int32_t _numVectors, int32_t _docBits)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            // Memory layout : [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            quantizedVectorBytes(
                validateQvbForDocBits(
                    _oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t)),
                    validateDocBits(_docBits))),
            // validateDocBits above already rejected _docBits ∉ {1,2,4}, so:
            //   - docBits / 0 division in `planeBytes` is unreachable
            //   - `1 << _docBits` signed-shift UB (would need _docBits > 30) is unreachable
            //   - silent wrong reconstruction for _docBits ∈ {3,5,6,7,...} is unreachable
            // Fields below are computed only when validation has passed.
            docBits(_docBits),
            planeBytes(quantizedVectorBytes / _docBits),
            dataScale(1.0f / static_cast<float>((1 << _docBits) - 1)),
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

        // Popcount of (a AND b) over `planeBytes` bytes. Uses std::memcpy for 8-byte word loads so
        // it is safe regardless of plane alignment (planeBytes need not be a multiple of 8 for B>1),
        // with a byte remainder loop for the trailing bytes.
        static inline uint64_t popcountAndPlane(const uint8_t* a, const uint8_t* b, const uint64_t planeBytes) {
            const uint64_t words = planeBytes >> 3;
            uint64_t pc = 0;
            for (uint64_t w = 0; w < words; ++w) {
                uint64_t wa, wb;
                std::memcpy(&wa, a + w * 8, sizeof(uint64_t));
                std::memcpy(&wb, b + w * 8, sizeof(uint64_t));
                pc += __builtin_popcountll(wa & wb);
            }
            for (uint64_t r = words * 8; r < planeBytes; ++r) {
                pc += __builtin_popcount((a[r] & b[r]) & 0xFF);
            }
            return pc;
        }

        // Byte-wise dot product over Lucene's PACKED_NIBBLE doc layout.
        // packed[i] high nibble = element i, low nibble = element (packedBytes + i).
        // Mirrors Lucene's VectorUtil.int4DotProductBothPacked. For each byte we extract two
        // 4-bit values per side and accumulate two products. The compiler's auto-vectorizer
        // turns this into wide 16-bit multiplies on x86 (PMULLW) and ARM (NEON MUL).
        //
        // WHY NOT a bit-plane popcount for B=4? PACKED_NIBBLE bytes are NOT bit planes — each
        // byte carries two elements at four different bit-positions each. popcount-AND on that
        // layout mixes place values (1, 2, 4, 8) into a uniform count and destroys the dot
        // product.
        static inline uint64_t bothPackedNibbleDp(const uint8_t* a, const uint8_t* b, const uint64_t packedBytes) {
            uint64_t total = 0;
            for (uint64_t i = 0; i < packedBytes; ++i) {
                const uint32_t aLo = a[i] & 0x0Fu;
                const uint32_t aHi = (a[i] >> 4) & 0x0Fu;
                const uint32_t bLo = b[i] & 0x0Fu;
                const uint32_t bHi = (b[i] >> 4) & 0x0Fu;
                total += aLo * bLo + aHi * bHi;
            }
            return total;
        }

        // Multi-bit dot product between two quantized codes. The kernel shape depends on docBits:
        //   B=1: single popcount(a AND b)                          — bit-plane (1 plane)
        //   B=2: 2x2 popcount-AND-shift double sum across planes   — bit-plane (2 planes)
        //   B=4: byte-wise nibble multiply-accumulate              — Lucene's PACKED_NIBBLE layout
        //
        // Performance trade-off: for B=4 the bit-plane formulation would need 16 popcount calls
        // per distance (and a Java-side repack, since Lucene stores PACKED_NIBBLE not bit planes),
        // while the byte-wise multiply matches Lucene's int4DotProductBothPacked and avoids both.
        // The math is the same (Σ A_n B_n with A_n, B_n in [0, 2^B - 1]); only the byte layout differs.
        uint64_t multiBitDp(const uint8_t* a, const uint8_t* b) const {
            if (docBits == 1) {
                return popcountAndPlane(a, b, planeBytes);
            }
            if (docBits == 4) {
                return bothPackedNibbleDp(a, b, quantizedVectorBytes);
            }
            // docBits == 2: bit-plane popcount path.
            uint64_t dp = 0;
            for (int32_t i = 0; i < docBits; ++i) {
                const uint8_t* pa = a + (uint64_t) i * planeBytes;
                for (int32_t j = 0; j < docBits; ++j) {
                    const uint8_t* pb = b + (uint64_t) j * planeBytes;
                    dp += popcountAndPlane(pa, pb, planeBytes) << (i + j);
                }
            }
            return dp;
        }

        void set_query(const float* x) final {
            // The query pointer comes from FaissSQFlat::quantizedVectorsAndCorrectionFactors
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
            // Scale (upper - lower) by 1/(2^docBits - 1) so the reconstruction delta matches the
            // quantization level range [0, 2^docBits - 1]. For docBits == 1 this is a no-op.
            intervalLength *= dataScale;
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
                // Negate: Faiss HNSW always minimizes distance (CMax comparator).
                // For IP, higher score = more similar, so we negate so that
                // minimizing -score = maximizing score.
                return -score;
            } else {
                // L2: squared distance = ||q||² + ||x||² - 2*dot(q,x)
                // additionalCorrection values carry the squared norms of centroid-centered vectors.
                return queryAdditional + additional - 2.0f * score;
            }
        }

        /// compute distance of vector i to current query
        float operator()(faiss::idx_t i) final {
            const uint8_t* target = data + i * oneElementByteSize;
            const uint64_t dp = multiBitDp(query, target);
            return scoringSecondPart(target, static_cast<float>(dp));
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

            dis0 = scoringSecondPart(target1, static_cast<float>(multiBitDp(query, target1)));
            dis1 = scoringSecondPart(target2, static_cast<float>(multiBitDp(query, target2)));
            dis2 = scoringSecondPart(target3, static_cast<float>(multiBitDp(query, target3)));
            dis3 = scoringSecondPart(target4, static_cast<float>(multiBitDp(query, target4)));
        }

        /// compute distance between two stored vectors
        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
            const uint8_t* target1 = data + i * oneElementByteSize;
            const uint8_t* target2 = data + j * oneElementByteSize;

            const uint64_t dp = multiBitDp(target1, target2);

            // Get correction factors
            float ax, lx, additional, x1;
            setCorrectionFactors(target1, ax, lx, additional, x1);

            float az, lz, additionalz, z1;
            setCorrectionFactors(target2, az, lz, additionalz, z1);

            // Scoring
            float score = ax * az * dimension
                   + az * lx * x1
                   + ax * lz * z1
                   + lx * lz * static_cast<float>(dp);

            if constexpr (IsMaxIP) {
                // Negate: Faiss HNSW always minimizes distance (CMax comparator).
                // For IP, higher score = more similar, so we negate so that
                // minimizing -score = maximizing score.
                score += additional + additionalz - centroidDp;
                return -score;
            } else {
                return additional + additionalz - 2 * score;
            }
        }
    };

    struct FaissSQFlat final : faiss::IndexBinary {
        int64_t numVectors;
        int32_t quantizedVectorBytes;
        float centroidDp;
        int32_t oneElementSize;
        // For safely casting uint8_t* to float*, we should enforce 8-byte alignment for the vector.
        std::vector<uint8_t, knn_jni::NBytesAlignedAllocator<uint8_t, 8>> quantizedVectorsAndCorrectionFactors;
        int32_t dimension;
        // Document bit width (1, 2, or 4). quantizedVectorBytes == docBits * binaryCodeBytes.
        int32_t docBits;

        FaissSQFlat(int64_t _numVectors, int32_t _quantizedVectorBytes, float _centroidDp, int32_t _dimension, faiss::MetricType _metric, int32_t _docBits)
          : faiss::IndexBinary(_dimension, _metric, true),
            numVectors(_numVectors),
            quantizedVectorBytes(_quantizedVectorBytes),
            centroidDp(_centroidDp),
            oneElementSize(_quantizedVectorBytes + 3 * sizeof(float) + sizeof(int32_t)),
            // Pre allocate vector storage space
            quantizedVectorsAndCorrectionFactors(_numVectors * oneElementSize),
            dimension(_dimension),
            docBits(_docBits) {

            // Just changing the size, not shrinking, thus allocated memory capacity remains the same.
            // This is to avoid reallocations when adding elements later on since we know the exact required memory space upfront.
            quantizedVectorsAndCorrectionFactors.resize(0);
            // Rewriting code_size to the full element size so that hnsw_add_vertices
            // strides correctly through the packed buffer when computing:
            //   x + (pt_id - n0) * index_hnsw.code_size
            // Memory layout per element:
            // [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            code_size = oneElementSize;
        }

        faiss::DistanceComputer* get_distance_computer() const {
            // When quantizedVectorBytes is a multiple of 8, oneElementSize is also a
            // multiple of 8 (quantizedVectorBytes + 16 bytes of correction factors),
            // so element starts are 8-byte aligned and we can use the fast
            // reinterpret_cast<uint64_t*> path. Otherwise we fall back to memcpy
            // with a byte remainder loop for the trailing bytes.
            const bool aligned = (oneElementSize % 8) == 0;
            if (metric_type == faiss::MetricType::METRIC_L2) {
                if (aligned) {
                    return new FaissSQDistanceComputer<false, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                } else {
                    return new FaissSQDistanceComputer<false, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                }
            } else if (metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                if (aligned) {
                    return new FaissSQDistanceComputer<true, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                } else {
                    return new FaissSQDistanceComputer<true, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
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
            throw std::runtime_error("FaissSQFlat does not support search");
        }

        void reset() final {
            throw std::runtime_error("FaissSQFlat does not support reset");
        }

        void merge_from(faiss::IndexBinary& otherIndex, faiss::idx_t add_id = 0) final {
            throw std::runtime_error("FaissSQFlat does not support merge_from");
        };

        void add(faiss::idx_t n, const uint8_t* x) final {
            // We only increase ntotal here, as it does not actually own the binary quantized vectors.
            // They are in a separate files.
            ntotal += n;
        }
    };

}

#endif //KNNPLUGIN_JNI_FAISS_SQ_FLAT_H
