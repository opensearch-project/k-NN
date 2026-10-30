/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

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
#include <string>
#include <iostream>

// SIMD kernels for the multi-bit distance computer. They are compiled with function-level target
// attributes and selected at runtime, so these headers add no global ISA requirement.
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

// AT_HWCAP bit for the Armv8.2 dot-product instructions (asm/hwcap.h); defined here in case the
// libc headers predate it.
#if defined(HWCAP_ASIMDDP)
#define KNN_SQ_HWCAP_ASIMDDP HWCAP_ASIMDDP
#else
#define KNN_SQ_HWCAP_ASIMDDP (1UL << 20)
#endif

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

    // Validates docBits for the 1-bit specialization below. FaissSQOneBitDistanceComputer
    // hard-codes the single-plane popcount kernel, so anything other than 1 must be routed to
    // FaissSQDistanceComputer. FaissSQFlat::get_distance_computer already does that routing; this
    // guard exists so a future caller cannot silently construct the wrong computer.
    static inline int32_t validateOneBitDocBits(int32_t bits) {
        if (validateDocBits(bits) != 1) {
            throw std::runtime_error(
                "FaissSQOneBitDistanceComputer: docBits=" + std::to_string(bits)
                + " is not 1. Use FaissSQDistanceComputer for docBits 2 and 4."
            );
        }
        return bits;
    }

    // Distance computer specialized for docBits == 1 (compression 32x), the overwhelmingly common
    // configuration.
    //
    // WHY A SEPARATE STRUCT? The multi-bit FaissSQDistanceComputer below has to decide the kernel
    // shape (1 plane / 2x2 planes / packed nibbles) from the runtime `docBits` field, which costs a
    // branch plus a non-inlinable helper call on every single distance, forces memcpy word loads
    // even when the data is known to be 8-byte aligned, and prevents the interleaved 4-target loop
    // in distances_batch_4 (one query-word load feeding four popcounts). Measured against the
    // 1-bit-only code it replaced, that cost 13-18% on symmetric_dis and up to 60% on
    // distances_batch_4. Splitting the 1-bit path out means the whole kernel is chosen at
    // construction time (once, in FaissSQFlat::get_distance_computer) and the hot loops below
    // contain no docBits logic at all. The bodies are intentionally kept identical to the
    // single-bit implementation that predates multi-bit support — for docBits == 1 the
    // reconstruction scale 1/(2^docBits - 1) is 1, so there is no interval scaling to apply and
    // the arithmetic is unchanged.
    template <bool IsMaxIP, bool IsBytesMultipleOf8>
    struct FaissSQOneBitDistanceComputer final : faiss::DistanceComputer {
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

        FaissSQOneBitDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp, int32_t _dimension, int32_t _numVectors, int32_t _docBits)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            // Memory layout : [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            // Validation runs in the initializer list, same as in FaissSQDistanceComputer, so an
            // unsupported docBits throws before any field is used.
            quantizedVectorBytes(
                validateQvbForDocBits(
                    _oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t)),
                    validateOneBitDocBits(_docBits))),
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
            // No dataScale multiply here: for docBits == 1 the scale is exactly 1.
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
            const uint64_t words = quantizedVectorBytes >> 3; // divide by 8
            uint32_t dp = 0;

            if constexpr (IsBytesMultipleOf8) {
                const auto* q = reinterpret_cast<const uint64_t*>(query);
                const auto* t = reinterpret_cast<const uint64_t*>(target);
                for (size_t j = 0; j < words; ++j) {
                    dp += __builtin_popcountll(q[j] & t[j]);
                }
            } else {
                for (size_t j = 0; j < words; ++j) {
                    uint64_t queryWord, targetWord;
                    std::memcpy(&queryWord, query + j * 8, sizeof(uint64_t));
                    std::memcpy(&targetWord, target + j * 8, sizeof(uint64_t));
                    dp += __builtin_popcountll(queryWord & targetWord);
                }
                // Remainder bytes that don't fill a full 8-byte word
                const uint64_t remainStart = words * 8;
                for (uint64_t r = remainStart; r < quantizedVectorBytes; ++r) {
                    dp += __builtin_popcount((query[r] & target[r]) & 0xFF);
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

            // Interleaved over the four targets on purpose: the query word is loaded once per
            // iteration and feeds four popcounts, which is what makes the batched call cheaper
            // than four independent operator() calls.
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
                // Remainder bytes that don't fill a full 8-byte word
                const uint64_t remainStart = words * 8;
                for (uint64_t r = remainStart; r < quantizedVectorBytes; ++r) {
                    const uint8_t qb = query[r];
                    dp1 += __builtin_popcount((qb & target1[r]) & 0xFF);
                    dp2 += __builtin_popcount((qb & target2[r]) & 0xFF);
                    dp3 += __builtin_popcount((qb & target3[r]) & 0xFF);
                    dp4 += __builtin_popcount((qb & target4[r]) & 0xFF);
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
                for (size_t k = 0; k < words; ++k) {
                    uint64_t w1, w2;
                    std::memcpy(&w1, target1 + k * 8, sizeof(uint64_t));
                    std::memcpy(&w2, target2 + k * 8, sizeof(uint64_t));
                    dp += __builtin_popcountll(w1 & w2);
                }
                // Remainder bytes that don't fill a full 8-byte word
                const uint64_t remainStart = words * 8;
                for (uint64_t r = remainStart; r < quantizedVectorBytes; ++r) {
                    dp += __builtin_popcount((target1[r] & target2[r]) & 0xFF);
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


    // ---------------------------------------------------------------------------------------------
    // Multi-bit (docBits 2 / 4) dot-product kernels
    // ---------------------------------------------------------------------------------------------
    //
    // Every kernel returns the exact integer dot product Σ A_n * B_n of two quantized codes, so all
    // implementations are bit-identical to each other (and to the previous uint64_t scalar code):
    // the result is converted to float exactly once, by the caller, in scoringSecondPart.
    //
    //   B=2: bit-plane layout, two planes of planeBytes each.
    //        dp = pc(a0&b0) + 2 * (pc(a0&b1) + pc(a1&b0)) + 4 * pc(a1&b1)
    //        (the 2x2 popcount-AND-shift double sum Σ_{i,j} pc(a_i & b_j) << (i + j)).
    //   B=4: Lucene PACKED_NIBBLE layout (VectorUtil.int4DotProductBothPacked), every byte carries
    //        two 4-bit elements: dp = Σ_i lo(a_i) * lo(b_i) + hi(a_i) * hi(b_i).
    //   B=1: single plane popcount. Only reachable through direct construction of
    //        FaissSQDistanceComputer (FaissSQFlat routes 1-bit to FaissSQOneBitDistanceComputer).
    //
    // ISA selection. The JNI libraries are built without ISA flags (no -mavx2 / -mpopcnt /
    // -march=...+dotprod), so the SIMD kernels are compiled with function-level
    // __attribute__((target(...))) and chosen ONCE per distance computer from a cached runtime CPU
    // probe. The hot loops therefore contain no docBits or ISA branches; each call is one
    // indirect call into the selected kernel. A portable scalar kernel is always available.
    //
    //   x86-64 : B=4 AVX2 (vpmaddubsw + vpmaddwd) if the CPU has AVX2, else scalar.
    //            B=2 POPCNT word loop if the CPU has POPCNT, else scalar (libgcc popcount).
    //   aarch64: B=4 NEON dot product (udot) if HWCAP_ASIMDDP (Graviton2/3/4), else baseline
    //            NEON (umull/umlal + uadalp). B=2 NEON vcnt. NEON itself is baseline on aarch64.
    //
    // Accumulator bounds. Every kernel accumulates in 32-bit lanes. Per code byte the dot product
    // grows by at most 2 * 15 * 15 = 450 (B=4) or 8 * (1 + 2 + 2 + 4) / 2 = 36 (B=2, per byte of
    // the 2*planeBytes code), so qvb <= kMaxMultiBitQuantizedBytes keeps every partial and final
    // sum below 2^31 (the AVX2 kernel uses signed epi32 lanes). The constructor enforces it.
    static constexpr uint64_t kMaxMultiBitQuantizedBytes = 0x7FFFFFFFull / 450ull;  // ~4.77M bytes

    // Kernel implementations. `Auto` picks the fastest one supported by the running CPU.
    enum class SQMultiBitImpl : int32_t {
        Auto = 0,
        Scalar,
        X86Popcnt,    // B=2 only
        X86Avx2,      // B=4 only
        Neon,         // B=2 (vcnt) and B=4 (umull/umlal)
        NeonDotProd,  // B=4 only (udot)
    };

    static inline const char* sqMultiBitImplName(SQMultiBitImpl impl) {
        switch (impl) {
            case SQMultiBitImpl::Auto: return "auto";
            case SQMultiBitImpl::Scalar: return "scalar";
            case SQMultiBitImpl::X86Popcnt: return "x86-popcnt";
            case SQMultiBitImpl::X86Avx2: return "x86-avx2";
            case SQMultiBitImpl::Neon: return "neon";
            case SQMultiBitImpl::NeonDotProd: return "neon-dotprod";
        }
        return "unknown";
    }

    // `n` is the kernel length in bytes: planeBytes for B=1/B=2, quantizedVectorBytes for B=4.
    using SQDpFn = uint32_t (*)(const uint8_t* a, const uint8_t* b, uint64_t n);
    using SQDp4Fn = void (*)(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                             const uint8_t* t2, const uint8_t* t3, uint64_t n, uint32_t* out);

    struct SQCpuFeatures {
        bool popcnt = false;
        bool avx2 = false;
        bool neonDotProd = false;
    };

    // Probed once per process (thread-safe function-local static).
    static inline const SQCpuFeatures& sqCpuFeatures() {
        static const SQCpuFeatures features = [] {
            SQCpuFeatures f;
#if defined(__x86_64__) && defined(__GNUC__)
            __builtin_cpu_init();
            f.popcnt = __builtin_cpu_supports("popcnt");
            // Also requires OS support for the YMM state; libgcc checks XGETBV for "avx2".
            f.avx2 = __builtin_cpu_supports("avx2");
#elif defined(__aarch64__) && defined(__linux__)
            f.neonDotProd = (getauxval(AT_HWCAP) & KNN_SQ_HWCAP_ASIMDDP) != 0;
#endif
            return f;
        }();
        return features;
    }

    static inline uint64_t sqLoadWord(const uint8_t* p) {
        uint64_t w;
        std::memcpy(&w, p, sizeof(uint64_t));
        return w;
    }

    // ---- B=1 (direct construction only) ----

    static inline uint32_t sqOneBitDpScalar(const uint8_t* a, const uint8_t* b, uint64_t n) {
        const uint64_t words = n >> 3;
        uint32_t pc = 0;
        for (uint64_t w = 0; w < words; ++w) {
            pc += __builtin_popcountll(sqLoadWord(a + w * 8) & sqLoadWord(b + w * 8));
        }
        for (uint64_t r = words * 8; r < n; ++r) {
            pc += __builtin_popcount((a[r] & b[r]) & 0xFF);
        }
        return pc;
    }

    static inline void sqOneBitDp4Scalar(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                                         const uint8_t* t2, const uint8_t* t3, uint64_t n, uint32_t* out) {
        out[0] = sqOneBitDpScalar(q, t0, n);
        out[1] = sqOneBitDpScalar(q, t1, n);
        out[2] = sqOneBitDpScalar(q, t2, n);
        out[3] = sqOneBitDpScalar(q, t3, n);
    }

    // ---- B=2: bit planes ----

    // Shared body for the scalar and POPCNT kernels; always_inline so that __builtin_popcountll
    // expands to the POPCNT instruction inside the target("popcnt") wrappers. Planes are read
    // with memcpy word loads, so planeBytes need not be a multiple of 8.
    __attribute__((always_inline)) static inline uint32_t sqTwoBitDpBody(
            const uint8_t* a0, const uint8_t* a1, const uint8_t* b0, const uint8_t* b1, uint64_t len) {
        const uint64_t words = len >> 3;
        uint32_t pc00 = 0, pcCross = 0, pc11 = 0;
        for (uint64_t w = 0; w < words; ++w) {
            const uint64_t wa0 = sqLoadWord(a0 + w * 8), wa1 = sqLoadWord(a1 + w * 8);
            const uint64_t wb0 = sqLoadWord(b0 + w * 8), wb1 = sqLoadWord(b1 + w * 8);
            pc00 += __builtin_popcountll(wa0 & wb0);
            pcCross += __builtin_popcountll(wa0 & wb1) + __builtin_popcountll(wa1 & wb0);
            pc11 += __builtin_popcountll(wa1 & wb1);
        }
        for (uint64_t r = words * 8; r < len; ++r) {
            pc00 += __builtin_popcount(a0[r] & b0[r]);
            pcCross += __builtin_popcount(a0[r] & b1[r]) + __builtin_popcount(a1[r] & b0[r]);
            pc11 += __builtin_popcount(a1[r] & b1[r]);
        }
        return pc00 + (pcCross << 1) + (pc11 << 2);
    }

    // Four targets against one query; each query word is loaded once.
    __attribute__((always_inline)) static inline void sqTwoBitDp4Body(
            const uint8_t* q, const uint8_t* const* t, uint64_t planeBytes, uint32_t* out) {
        const uint8_t* q1 = q + planeBytes;
        const uint64_t words = planeBytes >> 3;
        uint32_t pc00[4] = {0, 0, 0, 0}, pcCross[4] = {0, 0, 0, 0}, pc11[4] = {0, 0, 0, 0};
        for (uint64_t w = 0; w < words; ++w) {
            const uint64_t wq0 = sqLoadWord(q + w * 8), wq1 = sqLoadWord(q1 + w * 8);
            for (int k = 0; k < 4; ++k) {
                const uint64_t wt0 = sqLoadWord(t[k] + w * 8), wt1 = sqLoadWord(t[k] + planeBytes + w * 8);
                pc00[k] += __builtin_popcountll(wq0 & wt0);
                pcCross[k] += __builtin_popcountll(wq0 & wt1) + __builtin_popcountll(wq1 & wt0);
                pc11[k] += __builtin_popcountll(wq1 & wt1);
            }
        }
        for (int k = 0; k < 4; ++k) {
            const uint64_t r = words * 8;
            out[k] = pc00[k] + (pcCross[k] << 1) + (pc11[k] << 2)
                   + sqTwoBitDpBody(q + r, q1 + r, t[k] + r, t[k] + planeBytes + r, planeBytes - r);
        }
    }

    static inline uint32_t sqTwoBitDpScalar(const uint8_t* a, const uint8_t* b, uint64_t planeBytes) {
        return sqTwoBitDpBody(a, a + planeBytes, b, b + planeBytes, planeBytes);
    }

    static inline void sqTwoBitDp4Scalar(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                                         const uint8_t* t2, const uint8_t* t3, uint64_t planeBytes, uint32_t* out) {
        const uint8_t* t[4] = {t0, t1, t2, t3};
        sqTwoBitDp4Body(q, t, planeBytes, out);
    }

    // ---- B=4: packed nibbles ----

    // Scalar PACKED_NIBBLE dot product. Products are formed in 16 bits (each byte contributes at
    // most 450) and accumulated in 32 bits; with a 64-bit accumulator the auto-vectorizer had to
    // widen to 64-bit lanes (this alone is ~1.5x at -O3 on x86).
    static inline uint32_t sqNibbleDpScalar(const uint8_t* a, const uint8_t* b, uint64_t n) {
        uint32_t total = 0;
        for (uint64_t i = 0; i < n; ++i) {
            const uint16_t aLo = a[i] & 0x0Fu, aHi = a[i] >> 4;
            const uint16_t bLo = b[i] & 0x0Fu, bHi = b[i] >> 4;
            total += static_cast<uint16_t>(aLo * bLo + aHi * bHi);
        }
        return total;
    }

    static inline void sqNibbleDp4Scalar(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                                         const uint8_t* t2, const uint8_t* t3, uint64_t n, uint32_t* out) {
        out[0] = sqNibbleDpScalar(q, t0, n);
        out[1] = sqNibbleDpScalar(q, t1, n);
        out[2] = sqNibbleDpScalar(q, t2, n);
        out[3] = sqNibbleDpScalar(q, t3, n);
    }

#if defined(__x86_64__) && defined(__GNUC__)
    __attribute__((target("popcnt"))) static inline uint32_t sqTwoBitDpPopcnt(
            const uint8_t* a, const uint8_t* b, uint64_t planeBytes) {
        return sqTwoBitDpBody(a, a + planeBytes, b, b + planeBytes, planeBytes);
    }

    __attribute__((target("popcnt"))) static inline void sqTwoBitDp4Popcnt(
            const uint8_t* q, const uint8_t* t0, const uint8_t* t1, const uint8_t* t2, const uint8_t* t3,
            uint64_t planeBytes, uint32_t* out) {
        const uint8_t* t[4] = {t0, t1, t2, t3};
        sqTwoBitDp4Body(q, t, planeBytes, out);
    }

    __attribute__((target("avx2"))) static inline uint32_t sqHsumEpi32Avx2(__m256i v) {
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(s));
    }

    // Per 32 code bytes: split each side into low/high nibbles (values 0..15), then
    // vpmaddubsw multiplies u8 x s8 and adds adjacent pairs into int16. The nibbles are <= 15, so
    // the signed interpretation of the second operand is exact and each int16 is <= 2*225; the
    // lo and hi sums add to <= 900 (no saturation). vpmaddwd against 1s widens to int32 lanes.
    __attribute__((target("avx2"))) static inline __m256i sqNibbleBlockAvx2(
            __m256i aLo, __m256i aHi, const uint8_t* b, __m256i mask, __m256i ones) {
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        const __m256i bLo = _mm256_and_si256(vb, mask);
        const __m256i bHi = _mm256_and_si256(_mm256_srli_epi16(vb, 4), mask);
        const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(aLo, bLo), _mm256_maddubs_epi16(aHi, bHi));
        return _mm256_madd_epi16(p, ones);
    }

    __attribute__((target("avx2"))) static inline uint32_t sqNibbleDpAvx2(const uint8_t* a, const uint8_t* b, uint64_t n) {
        const __m256i mask = _mm256_set1_epi8(0x0F);
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i acc = _mm256_setzero_si256();
        uint64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i aLo = _mm256_and_si256(va, mask);
            const __m256i aHi = _mm256_and_si256(_mm256_srli_epi16(va, 4), mask);
            acc = _mm256_add_epi32(acc, sqNibbleBlockAvx2(aLo, aHi, b + i, mask, ones));
        }
        return sqHsumEpi32Avx2(acc) + sqNibbleDpScalar(a + i, b + i, n - i);
    }

    __attribute__((target("avx2"))) static inline void sqNibbleDp4Avx2(
            const uint8_t* q, const uint8_t* t0, const uint8_t* t1, const uint8_t* t2, const uint8_t* t3,
            uint64_t n, uint32_t* out) {
        const __m256i mask = _mm256_set1_epi8(0x0F);
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        uint64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i vq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
            const __m256i qLo = _mm256_and_si256(vq, mask);
            const __m256i qHi = _mm256_and_si256(_mm256_srli_epi16(vq, 4), mask);
            acc0 = _mm256_add_epi32(acc0, sqNibbleBlockAvx2(qLo, qHi, t0 + i, mask, ones));
            acc1 = _mm256_add_epi32(acc1, sqNibbleBlockAvx2(qLo, qHi, t1 + i, mask, ones));
            acc2 = _mm256_add_epi32(acc2, sqNibbleBlockAvx2(qLo, qHi, t2 + i, mask, ones));
            acc3 = _mm256_add_epi32(acc3, sqNibbleBlockAvx2(qLo, qHi, t3 + i, mask, ones));
        }
        out[0] = sqHsumEpi32Avx2(acc0) + sqNibbleDpScalar(q + i, t0 + i, n - i);
        out[1] = sqHsumEpi32Avx2(acc1) + sqNibbleDpScalar(q + i, t1 + i, n - i);
        out[2] = sqHsumEpi32Avx2(acc2) + sqNibbleDpScalar(q + i, t2 + i, n - i);
        out[3] = sqHsumEpi32Avx2(acc3) + sqNibbleDpScalar(q + i, t3 + i, n - i);
    }
#endif  // __x86_64__

#if defined(__aarch64__)
    // B=2 per 16 bytes of each plane: vcnt gives per-byte popcounts (<= 8), combined per byte as
    // c00 + 2*(c01 + c10) + 4*c11 <= 72, pairwise-accumulated into u16 lanes (<= 144 per
    // iteration). The u16 accumulator is flushed into u32 every kTwoBitNeonBlock iterations so
    // it never exceeds 255 * 144 < 65536.
    static constexpr uint64_t kTwoBitNeonBlock = 255;

    static inline uint8x16_t sqTwoBitWeightedCountNeon(uint8x16_t a0, uint8x16_t a1, uint8x16_t b0, uint8x16_t b1) {
        const uint8x16_t c00 = vcntq_u8(vandq_u8(a0, b0));
        const uint8x16_t cCross = vaddq_u8(vcntq_u8(vandq_u8(a0, b1)), vcntq_u8(vandq_u8(a1, b0)));
        const uint8x16_t c11 = vcntq_u8(vandq_u8(a1, b1));
        return vaddq_u8(vaddq_u8(c00, vshlq_n_u8(cCross, 1)), vshlq_n_u8(c11, 2));
    }

    static inline uint32_t sqTwoBitDpNeon(const uint8_t* a, const uint8_t* b, uint64_t planeBytes) {
        const uint8_t* a1 = a + planeBytes;
        const uint8_t* b1 = b + planeBytes;
        uint32x4_t acc32 = vdupq_n_u32(0);
        uint64_t i = 0;
        while (i + 16 <= planeBytes) {
            uint16x8_t acc16 = vdupq_n_u16(0);
            for (uint64_t it = 0; it < kTwoBitNeonBlock && i + 16 <= planeBytes; ++it, i += 16) {
                acc16 = vpadalq_u8(acc16, sqTwoBitWeightedCountNeon(
                    vld1q_u8(a + i), vld1q_u8(a1 + i), vld1q_u8(b + i), vld1q_u8(b1 + i)));
            }
            acc32 = vpadalq_u16(acc32, acc16);
        }
        return vaddvq_u32(acc32) + sqTwoBitDpBody(a + i, a1 + i, b + i, b1 + i, planeBytes - i);
    }

    static inline void sqTwoBitDp4Neon(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                                       const uint8_t* t2, const uint8_t* t3, uint64_t planeBytes, uint32_t* out) {
        const uint8_t* t[4] = {t0, t1, t2, t3};
        const uint8_t* q1 = q + planeBytes;
        uint32x4_t acc32[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
        uint64_t i = 0;
        while (i + 16 <= planeBytes) {
            uint16x8_t acc16[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
            for (uint64_t it = 0; it < kTwoBitNeonBlock && i + 16 <= planeBytes; ++it, i += 16) {
                const uint8x16_t vq0 = vld1q_u8(q + i), vq1 = vld1q_u8(q1 + i);
                for (int k = 0; k < 4; ++k) {
                    acc16[k] = vpadalq_u8(acc16[k], sqTwoBitWeightedCountNeon(
                        vq0, vq1, vld1q_u8(t[k] + i), vld1q_u8(t[k] + planeBytes + i)));
                }
            }
            for (int k = 0; k < 4; ++k) {
                acc32[k] = vpadalq_u16(acc32[k], acc16[k]);
            }
        }
        for (int k = 0; k < 4; ++k) {
            out[k] = vaddvq_u32(acc32[k])
                   + sqTwoBitDpBody(q + i, q1 + i, t[k] + i, t[k] + planeBytes + i, planeBytes - i);
        }
    }

    // B=4 baseline NEON per 16 bytes: umull/umlal accumulate the four 8x8 nibble products of a
    // lane pair into u16 (<= 4 * 225 = 900), then uadalp widens into u32.
    static inline uint16x8_t sqNibbleProductsNeon(uint8x16_t aLo, uint8x16_t aHi, uint8x16_t vb) {
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        const uint8x16_t bLo = vandq_u8(vb, mask);
        const uint8x16_t bHi = vshrq_n_u8(vb, 4);
        uint16x8_t p = vmull_u8(vget_low_u8(aLo), vget_low_u8(bLo));
        p = vmlal_u8(p, vget_low_u8(aHi), vget_low_u8(bHi));
        p = vmlal_high_u8(p, aLo, bLo);
        return vmlal_high_u8(p, aHi, bHi);
    }

    static inline uint32_t sqNibbleDpNeon(const uint8_t* a, const uint8_t* b, uint64_t n) {
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        uint32x4_t acc = vdupq_n_u32(0);
        uint64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t va = vld1q_u8(a + i);
            acc = vpadalq_u16(acc, sqNibbleProductsNeon(vandq_u8(va, mask), vshrq_n_u8(va, 4), vld1q_u8(b + i)));
        }
        return vaddvq_u32(acc) + sqNibbleDpScalar(a + i, b + i, n - i);
    }

    static inline void sqNibbleDp4Neon(const uint8_t* q, const uint8_t* t0, const uint8_t* t1,
                                       const uint8_t* t2, const uint8_t* t3, uint64_t n, uint32_t* out) {
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        uint32x4_t acc0 = vdupq_n_u32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        uint64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t vq = vld1q_u8(q + i);
            const uint8x16_t qLo = vandq_u8(vq, mask), qHi = vshrq_n_u8(vq, 4);
            acc0 = vpadalq_u16(acc0, sqNibbleProductsNeon(qLo, qHi, vld1q_u8(t0 + i)));
            acc1 = vpadalq_u16(acc1, sqNibbleProductsNeon(qLo, qHi, vld1q_u8(t1 + i)));
            acc2 = vpadalq_u16(acc2, sqNibbleProductsNeon(qLo, qHi, vld1q_u8(t2 + i)));
            acc3 = vpadalq_u16(acc3, sqNibbleProductsNeon(qLo, qHi, vld1q_u8(t3 + i)));
        }
        out[0] = vaddvq_u32(acc0) + sqNibbleDpScalar(q + i, t0 + i, n - i);
        out[1] = vaddvq_u32(acc1) + sqNibbleDpScalar(q + i, t1 + i, n - i);
        out[2] = vaddvq_u32(acc2) + sqNibbleDpScalar(q + i, t2 + i, n - i);
        out[3] = vaddvq_u32(acc3) + sqNibbleDpScalar(q + i, t3 + i, n - i);
    }

    // B=4 with the Armv8.2 dot-product extension: udot sums four u8 x u8 products into each u32
    // lane, so a 16-byte block is two udot instructions (low nibbles, high nibbles). Compiled for
    // armv8.2-a+dotprod at function level and only selected when HWCAP_ASIMDDP is set.
#define KNN_SQ_DOTPROD_TARGET __attribute__((target("arch=armv8.2-a+dotprod")))
    KNN_SQ_DOTPROD_TARGET static inline uint32_t sqNibbleDpNeonDotProd(const uint8_t* a, const uint8_t* b, uint64_t n) {
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        uint32x4_t accLo = vdupq_n_u32(0), accHi = vdupq_n_u32(0);
        uint64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t va = vld1q_u8(a + i), vb = vld1q_u8(b + i);
            accLo = vdotq_u32(accLo, vandq_u8(va, mask), vandq_u8(vb, mask));
            accHi = vdotq_u32(accHi, vshrq_n_u8(va, 4), vshrq_n_u8(vb, 4));
        }
        return vaddvq_u32(vaddq_u32(accLo, accHi)) + sqNibbleDpScalar(a + i, b + i, n - i);
    }

    KNN_SQ_DOTPROD_TARGET static inline void sqNibbleDp4NeonDotProd(
            const uint8_t* q, const uint8_t* t0, const uint8_t* t1, const uint8_t* t2, const uint8_t* t3,
            uint64_t n, uint32_t* out) {
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        uint32x4_t acc0 = vdupq_n_u32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        uint64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t vq = vld1q_u8(q + i);
            const uint8x16_t qLo = vandq_u8(vq, mask), qHi = vshrq_n_u8(vq, 4);
            const uint8x16_t v0 = vld1q_u8(t0 + i), v1 = vld1q_u8(t1 + i);
            const uint8x16_t v2 = vld1q_u8(t2 + i), v3 = vld1q_u8(t3 + i);
            acc0 = vdotq_u32(vdotq_u32(acc0, qLo, vandq_u8(v0, mask)), qHi, vshrq_n_u8(v0, 4));
            acc1 = vdotq_u32(vdotq_u32(acc1, qLo, vandq_u8(v1, mask)), qHi, vshrq_n_u8(v1, 4));
            acc2 = vdotq_u32(vdotq_u32(acc2, qLo, vandq_u8(v2, mask)), qHi, vshrq_n_u8(v2, 4));
            acc3 = vdotq_u32(vdotq_u32(acc3, qLo, vandq_u8(v3, mask)), qHi, vshrq_n_u8(v3, 4));
        }
        out[0] = vaddvq_u32(acc0) + sqNibbleDpScalar(q + i, t0 + i, n - i);
        out[1] = vaddvq_u32(acc1) + sqNibbleDpScalar(q + i, t1 + i, n - i);
        out[2] = vaddvq_u32(acc2) + sqNibbleDpScalar(q + i, t2 + i, n - i);
        out[3] = vaddvq_u32(acc3) + sqNibbleDpScalar(q + i, t3 + i, n - i);
    }
#undef KNN_SQ_DOTPROD_TARGET
#endif  // __aarch64__

    struct SQMultiBitKernel {
        SQMultiBitImpl impl;
        SQDpFn dp;
        SQDp4Fn dp4;
    };

    // Resolves (docBits, requested implementation) to a kernel. Returns false if the requested
    // implementation does not exist for this docBits / architecture or the CPU lacks the ISA
    // extension. `Auto` always succeeds.
    static inline bool resolveSQMultiBitKernel(int32_t bits, SQMultiBitImpl requested, SQMultiBitKernel& out) {
        const SQCpuFeatures& cpu = sqCpuFeatures();
        if (bits == 1) {
            if (requested != SQMultiBitImpl::Auto && requested != SQMultiBitImpl::Scalar) return false;
            out = {SQMultiBitImpl::Scalar, &sqOneBitDpScalar, &sqOneBitDp4Scalar};
            return true;
        }
        if (requested == SQMultiBitImpl::Auto) {
#if defined(__x86_64__) && defined(__GNUC__)
            requested = bits == 2 ? (cpu.popcnt ? SQMultiBitImpl::X86Popcnt : SQMultiBitImpl::Scalar)
                                  : (cpu.avx2 ? SQMultiBitImpl::X86Avx2 : SQMultiBitImpl::Scalar);
#elif defined(__aarch64__)
            requested = (bits == 4 && cpu.neonDotProd) ? SQMultiBitImpl::NeonDotProd : SQMultiBitImpl::Neon;
#else
            requested = SQMultiBitImpl::Scalar;
#endif
        }
        if (requested == SQMultiBitImpl::Scalar) {
            out = bits == 2 ? SQMultiBitKernel{requested, &sqTwoBitDpScalar, &sqTwoBitDp4Scalar}
                            : SQMultiBitKernel{requested, &sqNibbleDpScalar, &sqNibbleDp4Scalar};
            return true;
        }
#if defined(__x86_64__) && defined(__GNUC__)
        if (requested == SQMultiBitImpl::X86Popcnt && bits == 2 && cpu.popcnt) {
            out = {requested, &sqTwoBitDpPopcnt, &sqTwoBitDp4Popcnt};
            return true;
        }
        if (requested == SQMultiBitImpl::X86Avx2 && bits == 4 && cpu.avx2) {
            out = {requested, &sqNibbleDpAvx2, &sqNibbleDp4Avx2};
            return true;
        }
#endif
#if defined(__aarch64__)
        if (requested == SQMultiBitImpl::Neon) {
            out = bits == 2 ? SQMultiBitKernel{requested, &sqTwoBitDpNeon, &sqTwoBitDp4Neon}
                            : SQMultiBitKernel{requested, &sqNibbleDpNeon, &sqNibbleDp4Neon};
            return true;
        }
        if (requested == SQMultiBitImpl::NeonDotProd && bits == 4 && cpu.neonDotProd) {
            out = {requested, &sqNibbleDpNeonDotProd, &sqNibbleDp4NeonDotProd};
            return true;
        }
#endif
        (void) cpu;
        return false;
    }

    // Rejects codes long enough to overflow the 32-bit kernel accumulators (see
    // kMaxMultiBitQuantizedBytes). Real codes are far below the limit (D=16000 at B=4 is 8000 bytes).
    static inline uint64_t validateQvbForAccumulator(uint64_t qvb) {
        if (qvb > kMaxMultiBitQuantizedBytes) {
            throw std::runtime_error(
                "FaissSQDistanceComputer: quantizedVectorBytes=" + std::to_string(qvb)
                + " exceeds the supported maximum " + std::to_string(kMaxMultiBitQuantizedBytes)
            );
        }
        return qvb;
    }

    // Distance computer for multi-bit codes (docBits 2 and 4). docBits == 1 is served by
    // FaissSQOneBitDistanceComputer above (see the comment there); B=1 here is supported only for
    // direct construction (unit tests). The dot-product kernel — layout (bit planes vs packed
    // nibbles) and ISA — is resolved once in the constructor into `kernel`, so the distance entry
    // points below carry no docBits branch and make exactly one indirect kernel call each
    // (distances_batch_4 included: the four targets are processed in one interleaved pass).
    template <bool IsMaxIP, bool IsBytesMultipleOf8>
    struct FaissSQDistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const uint64_t quantizedVectorBytes;
        // Number of bits used to quantize each document dimension (1, 2, or 4).
        const int32_t docBits;
        // Byte length of a single bit plane. Only meaningful for the bit-plane popcount path
        // (B=1, B=2): the quantized code is docBits contiguous planes, each planeBytes long,
        // so quantizedVectorBytes == docBits * planeBytes. Unused for B=4 (PACKED_NIBBLE),
        // where the nibble kernels iterate over quantizedVectorBytes directly.
        const uint64_t planeBytes;
        // Length argument handed to the kernel: planeBytes for B=1/B=2, quantizedVectorBytes for B=4.
        const uint64_t kernelBytes;
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
        // Selected dot-product kernel; see resolveSQMultiBitKernel.
        SQMultiBitKernel kernel;

        FaissSQDistanceComputer(int32_t _oneElementByteSize, const void* _data, float _centroidDp, int32_t _dimension, int32_t _numVectors, int32_t _docBits)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            // Memory layout : [Quantized Vector | lowerInterval (float) | upperInterval (float) | additionalCorrection (float) | quantizedComponentSum (int)]
            quantizedVectorBytes(
                validateQvbForAccumulator(
                    validateQvbForDocBits(
                        _oneElementByteSize - (sizeof(float) * 3 + sizeof(int32_t)),
                        validateDocBits(_docBits)))),
            // validateDocBits above already rejected _docBits ∉ {1,2,4}, so:
            //   - docBits / 0 division in `planeBytes` is unreachable
            //   - `1 << _docBits` signed-shift UB (would need _docBits > 30) is unreachable
            //   - silent wrong reconstruction for _docBits ∈ {3,5,6,7,...} is unreachable
            // Fields below are computed only when validation has passed.
            docBits(_docBits),
            planeBytes(quantizedVectorBytes / _docBits),
            kernelBytes(_docBits == 4 ? quantizedVectorBytes : quantizedVectorBytes / _docBits),
            dataScale(1.0f / static_cast<float>((1 << _docBits) - 1)),
            data((const uint8_t*) _data),
            query(),
            centroidDp(_centroidDp),
            ay(),
            ly(),
            queryAdditional(),
            y1(),
            dimension(_dimension),
            numVectors(_numVectors),
            kernel() {
            resolveSQMultiBitKernel(docBits, SQMultiBitImpl::Auto, kernel);
        }

        // Overrides the automatically selected kernel (used by tests and benchmarks to exercise
        // every implementation). Returns false, leaving the current kernel in place, when the
        // implementation is unavailable for this docBits / architecture / CPU.
        bool setImplementation(SQMultiBitImpl impl) {
            SQMultiBitKernel k;
            if (!resolveSQMultiBitKernel(docBits, impl, k)) {
                return false;
            }
            kernel = k;
            return true;
        }

        SQMultiBitImpl implementation() const {
            return kernel.impl;
        }

        // Exact integer dot product between two quantized codes (see the kernel comment above).
        uint32_t multiBitDp(const uint8_t* a, const uint8_t* b) const {
            return kernel.dp(a, b, kernelBytes);
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
            const uint32_t dp = kernel.dp(query, target, kernelBytes);
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

            // One interleaved pass: each query block is loaded once and feeds all four targets.
            uint32_t dp[4];
            kernel.dp4(query, target1, target2, target3, target4, kernelBytes, dp);

            dis0 = scoringSecondPart(target1, static_cast<float>(dp[0]));
            dis1 = scoringSecondPart(target2, static_cast<float>(dp[1]));
            dis2 = scoringSecondPart(target3, static_cast<float>(dp[2]));
            dis3 = scoringSecondPart(target4, static_cast<float>(dp[3]));
        }

        /// compute distance between two stored vectors
        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
            const uint8_t* target1 = data + i * oneElementByteSize;
            const uint8_t* target2 = data + j * oneElementByteSize;

            const uint32_t dp = kernel.dp(target1, target2, kernelBytes);

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
        // 64-bit so that ordinal * oneElementSize cannot overflow once the storage exceeds 2 GiB.
        int64_t oneElementSize;
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

        // Returns a pointer to the `numElements` stored elements starting at ordinal `startOrd`.
        // Throws if the range is not within the elements stored so far, so a bad batch fails the
        // build with an exception instead of reading outside the buffer.
        const uint8_t* elementsAt(int64_t startOrd, int64_t numElements) const {
            const int64_t storedElements = static_cast<int64_t>(quantizedVectorsAndCorrectionFactors.size()) / oneElementSize;
            if (startOrd < 0 || numElements < 0 || startOrd + numElements > storedElements) {
                throw std::runtime_error(
                    "SQ elements [" + std::to_string(startOrd) + ", " + std::to_string(startOrd + numElements)
                    + ") are out of bounds of the " + std::to_string(storedElements) + " stored elements"
                );
            }
            return quantizedVectorsAndCorrectionFactors.data() + startOrd * oneElementSize;
        }

        faiss::DistanceComputer* get_distance_computer() const {
            // When quantizedVectorBytes is a multiple of 8, oneElementSize is also a
            // multiple of 8 (quantizedVectorBytes + 16 bytes of correction factors),
            // so element starts are 8-byte aligned and we can use the fast
            // reinterpret_cast<uint64_t*> path. Otherwise we fall back to memcpy
            // with a byte remainder loop for the trailing bytes.
            const bool aligned = (oneElementSize % 8) == 0;
            // Resolve 1-bit vs multi-bit here, once per distance computer, so that neither
            // kernel pays for the other's generality on every distance call. docBits is fixed
            // for the lifetime of the index, so this is the earliest and only place the
            // decision needs to be made.
            const bool oneBit = docBits == 1;
            if (metric_type == faiss::MetricType::METRIC_L2) {
                if (oneBit) {
                    if (aligned) {
                        return new FaissSQOneBitDistanceComputer<false, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                    } else {
                        return new FaissSQOneBitDistanceComputer<false, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                    }
                }
                if (aligned) {
                    return new FaissSQDistanceComputer<false, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                } else {
                    return new FaissSQDistanceComputer<false, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                }
            } else if (metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                if (oneBit) {
                    if (aligned) {
                        return new FaissSQOneBitDistanceComputer<true, true>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                    } else {
                        return new FaissSQOneBitDistanceComputer<true, false>(oneElementSize, quantizedVectorsAndCorrectionFactors.data(), centroidDp, dimension, numVectors, docBits);
                    }
                }
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
