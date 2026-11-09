/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_SQ_SEARCH_MULTIBIT_KERNELS_H
#define OPENSEARCH_KNN_SIMD_SQ_SEARCH_MULTIBIT_KERNELS_H

#include <cstdint>
#include <cstring>

#include "platform_defs.h"

// MinGW GCC cannot keep 32/64-byte AVX values on the 16-byte aligned Win64 stack, so Windows uses the scalar kernels,
// like the SQ indexing kernels in sq/faiss_sq_flat.h.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(_WIN32)
#include <immintrin.h>
#define KNN_SQ_SEARCH_X86 1
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#define KNN_SQ_SEARCH_ARM 1
#endif

namespace knn_jni::simd::similarity_function {

    // Code dot products of the 4-bit query against 2-bit (DIBIT_QUERY_NIBBLE) and 4-bit (PACKED_NIBBLE) Lucene document
    // codes, for HNSW search over LuceneSqFormat records. These are deliberately separate from the cluster ANN kernels of
    // the ISA files: they are written for scattered records (one query against four records per call, no streaming
    // prefetch) and, being selected at runtime from the CPU's features through function-level target attributes, they use
    // the best instructions the host has whichever simd library was loaded (so AVX-512 VNNI / VPOPCNTDQ on Ice Lake and
    // Zen 4, which load the plain avx512 library, and AVX2 on AVX2-only hosts).
    //
    //   2-bit: the query is 4 bit planes of `n` = stripe bytes (transposeHalfByte); the document is bit plane 0 then bit
    //          plane 1, `n` bytes each. dot = sum_p 2^p * (popcount(q_p & d_0) + 2 * popcount(q_p & d_1)), Lucene's
    //          int4DibitDotProduct.
    //   4-bit: the query is one 4-bit code per discretized dimension, 2 * `n` bytes; the document is `n` packed bytes.
    //          dot = sum_i q[i] * (d[i] >> 4) + q[i + n] * (d[i] & 0x0F), Lucene's int4DotProductSinglePacked.
    //
    // Every implementation computes the exact integer, so all of them return the same value. The sums are bounded by
    // 45 * dimension (2-bit) and 450 * (dimension / 2) (4-bit), far inside uint32 for any supported dimension. Tails are
    // either masked loads (AVX-512, which never fault on masked-off bytes) or scalar loops: no implementation reads past
    // the `n` (2-bit: 4n query, 2n document; 4-bit: 2n query, n document) bytes it is given.

    enum class SqSearchImpl : int32_t {
        Auto = 0,
        Scalar,
        X86Avx2,          // 2-bit vpshufb popcount, 4-bit vpmaddubsw + vpmaddwd (256-bit)
        X86Avx512Bw,      // the same on 512-bit with masked tails
        X86Avx512Popcnt,  // 2-bit only: vpopcntq (AVX512-VPOPCNTDQ)
        X86Avx512Vnni,    // 4-bit only: vpdpbusd (AVX512-VNNI)
        Neon,             // 2-bit vcnt, 4-bit umull + uadalp
        NeonDotProd,      // 4-bit only: udot
    };

    static inline const char* sqSearchImplName(const SqSearchImpl impl) {
        switch (impl) {
            case SqSearchImpl::Auto: return "auto";
            case SqSearchImpl::Scalar: return "scalar";
            case SqSearchImpl::X86Avx2: return "x86-avx2";
            case SqSearchImpl::X86Avx512Bw: return "x86-avx512bw";
            case SqSearchImpl::X86Avx512Popcnt: return "x86-avx512-vpopcntdq";
            case SqSearchImpl::X86Avx512Vnni: return "x86-avx512-vnni";
            case SqSearchImpl::Neon: return "neon";
            case SqSearchImpl::NeonDotProd: return "neon-dotprod";
        }
        return "unknown";
    }

    using SqSearchDpFn = uint32_t (*)(const uint8_t* q, const uint8_t* d, int32_t n);
    using SqSearchDp4Fn = void (*)(const uint8_t* q, const uint8_t* d0, const uint8_t* d1, const uint8_t* d2,
                                   const uint8_t* d3, int32_t n, uint32_t* out);
    using SqSearchDp8Fn = void (*)(const uint8_t* q, const uint8_t* const* d, int32_t n, uint32_t* out);

    struct SqSearchKernel {
        SqSearchImpl impl = SqSearchImpl::Scalar;
        SqSearchDpFn dp = nullptr;
        SqSearchDp4Fn dp4 = nullptr;
        // Eight records per call; only set where it measured faster than four (NEON dotprod 4-bit).
        SqSearchDp8Fn dp8 = nullptr;
    };

    // ---- Scalar (every platform) ----

    static inline uint32_t sqSearchDibitScalar(const uint8_t* q, const uint8_t* d, const int32_t n) {
        uint64_t acc0 = 0, acc1 = 0;
        for (int32_t p = 0; p < 4; ++p) {
            const uint8_t* plane = q + static_cast<int64_t>(p) * n;
            uint64_t c0 = 0, c1 = 0;
            for (int32_t i = 0; i < n; ++i) {
                c0 += static_cast<uint64_t>(__builtin_popcount(plane[i] & d[i]));
                c1 += static_cast<uint64_t>(__builtin_popcount(plane[i] & d[n + i]));
            }
            acc0 += c0 << p;
            acc1 += c1 << p;
        }
        return static_cast<uint32_t>(acc0 + (acc1 << 1));
    }

    static inline void sqSearchDibit4Scalar(const uint8_t* q, const uint8_t* d0, const uint8_t* d1, const uint8_t* d2,
                                            const uint8_t* d3, const int32_t n, uint32_t* out) {
        out[0] = sqSearchDibitScalar(q, d0, n);
        out[1] = sqSearchDibitScalar(q, d1, n);
        out[2] = sqSearchDibitScalar(q, d2, n);
        out[3] = sqSearchDibitScalar(q, d3, n);
    }

    static inline uint32_t sqSearchNibbleScalar(const uint8_t* q, const uint8_t* d, const int32_t n) {
        uint32_t sum = 0;
        for (int32_t i = 0; i < n; ++i) {
            sum += static_cast<uint32_t>(q[i]) * static_cast<uint32_t>(d[i] >> 4)
                 + static_cast<uint32_t>(q[n + i]) * static_cast<uint32_t>(d[i] & 0x0F);
        }
        return sum;
    }

    static inline void sqSearchNibble4Scalar(const uint8_t* q, const uint8_t* d0, const uint8_t* d1, const uint8_t* d2,
                                             const uint8_t* d3, const int32_t n, uint32_t* out) {
        out[0] = sqSearchNibbleScalar(q, d0, n);
        out[1] = sqSearchNibbleScalar(q, d1, n);
        out[2] = sqSearchNibbleScalar(q, d2, n);
        out[3] = sqSearchNibbleScalar(q, d3, n);
    }

    // The 1- and 4-record entry points (the SqSearchKernel function pointers) of a kernel template KERNEL<N>, carrying the
    // kernel's target attribute so the template inlines into them.
#define KNN_SQ_SEARCH_ENTRY_POINTS(TARGET, KERNEL)                                                                      \
    TARGET static uint32_t KERNEL##One(const uint8_t* q, const uint8_t* d, const int32_t n) {                          \
        uint32_t out;                                                                                                  \
        const uint8_t* one[1] = {d};                                                                                   \
        KERNEL<1>(q, one, n, &out);                                                                                     \
        return out;                                                                                                    \
    }                                                                                                                  \
    TARGET static void KERNEL##Four(const uint8_t* q, const uint8_t* d0, const uint8_t* d1, const uint8_t* d2,          \
                                    const uint8_t* d3, const int32_t n, uint32_t* out) {                               \
        const uint8_t* d[4] = {d0, d1, d2, d3};                                                                        \
        KERNEL<4>(q, d, n, out);                                                                                        \
    }
#define KNN_SQ_SEARCH_NO_TARGET

#if defined(KNN_SQ_SEARCH_X86)
#define KNN_SQ_SEARCH_AVX2 __attribute__((target("avx2,popcnt")))
#define KNN_SQ_SEARCH_AVX512BW __attribute__((target("avx512f,avx512bw")))
#define KNN_SQ_SEARCH_AVX512POPCNT __attribute__((target("avx512f,avx512bw,avx512vpopcntdq")))
#define KNN_SQ_SEARCH_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512vnni")))

    // ---- AVX2: 32 bytes per step, scalar tail ----

    // Per-byte popcount of v by the nibble lookup table (vpshufb).
    KNN_SQ_SEARCH_AVX2 static inline __m256i sqSearchPopcnt8Avx2(const __m256i v) {
        const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                             0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i low = _mm256_set1_epi8(0x0F);
        return _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v, low)),
                               _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), low)));
    }

    // popcount(q0&d) + 2 popcount(q1&d) + 4 popcount(q2&d) + 8 popcount(q3&d) per byte (<= 120), summed into 4 u64 lanes.
    KNN_SQ_SEARCH_AVX2 static inline __m256i sqSearchWeightedPopcntAvx2(const __m256i q0, const __m256i q1,
                                                                        const __m256i q2, const __m256i q3,
                                                                        const __m256i d) {
        __m256i w = sqSearchPopcnt8Avx2(_mm256_and_si256(q0, d));
        w = _mm256_add_epi8(w, _mm256_slli_epi16(sqSearchPopcnt8Avx2(_mm256_and_si256(q1, d)), 1));
        w = _mm256_add_epi8(w, _mm256_slli_epi16(sqSearchPopcnt8Avx2(_mm256_and_si256(q2, d)), 2));
        w = _mm256_add_epi8(w, _mm256_slli_epi16(sqSearchPopcnt8Avx2(_mm256_and_si256(q3, d)), 3));
        return _mm256_sad_epu8(w, _mm256_setzero_si256());
    }

    KNN_SQ_SEARCH_AVX2 static inline uint64_t sqSearchHsumEpi64Avx2(const __m256i v) {
        const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        return static_cast<uint64_t>(_mm_cvtsi128_si64(s)) + static_cast<uint64_t>(_mm_extract_epi64(s, 1));
    }

    KNN_SQ_SEARCH_AVX2 static inline uint32_t sqSearchHsumEpi32Avx2(const __m256i v) {
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(s));
    }

    // Scalar remainder of the 2-bit dot product over bytes [from, n).
    static inline uint64_t sqSearchDibitTail(const uint8_t* q, const uint8_t* d, const int32_t n, const int32_t from) {
        uint64_t acc = 0;
        for (int32_t i = from; i < n; ++i) {
            for (int32_t p = 0; p < 4; ++p) {
                const uint8_t qb = q[static_cast<int64_t>(p) * n + i];
                acc += static_cast<uint64_t>(__builtin_popcount(qb & d[i]) + 2 * __builtin_popcount(qb & d[n + i])) << p;
            }
        }
        return acc;
    }

    // Scalar remainder of the 4-bit dot product over bytes [from, n).
    static inline uint32_t sqSearchNibbleTail(const uint8_t* q, const uint8_t* d, const int32_t n, const int32_t from) {
        uint32_t sum = 0;
        for (int32_t i = from; i < n; ++i) {
            sum += static_cast<uint32_t>(q[i]) * static_cast<uint32_t>(d[i] >> 4)
                 + static_cast<uint32_t>(q[n + i]) * static_cast<uint32_t>(d[i] & 0x0F);
        }
        return sum;
    }

    template <int N>
    KNN_SQ_SEARCH_AVX2 static inline void sqSearchDibitAvx2(const uint8_t* q, const uint8_t* const* d, const int32_t n,
                                                            uint32_t* out) {
        __m256i acc0[N], acc1[N];
        for (int k = 0; k < N; ++k) {
            acc0[k] = _mm256_setzero_si256();
            acc1[k] = _mm256_setzero_si256();
        }
        int32_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
            const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + n + i));
            const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 2 * static_cast<int64_t>(n) + i));
            const __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 3 * static_cast<int64_t>(n) + i));
            for (int k = 0; k < N; ++k) {
                const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d[k] + i));
                const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d[k] + n + i));
                acc0[k] = _mm256_add_epi64(acc0[k], sqSearchWeightedPopcntAvx2(q0, q1, q2, q3, p0));
                acc1[k] = _mm256_add_epi64(acc1[k], sqSearchWeightedPopcntAvx2(q0, q1, q2, q3, p1));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(sqSearchHsumEpi64Avx2(acc0[k]) + (sqSearchHsumEpi64Avx2(acc1[k]) << 1)
                                           + sqSearchDibitTail(q, d[k], n, i));
        }
    }

    // hi * qHi + lo * qLo per byte pair into int16 (<= 2 * 225 each, <= 900 summed), then pairs of int16 into int32.
    KNN_SQ_SEARCH_AVX2 static inline __m256i sqSearchNibbleBlockAvx2(const __m256i qh, const __m256i ql, const __m256i p) {
        const __m256i low = _mm256_set1_epi8(0x0F);
        const __m256i lo = _mm256_and_si256(p, low);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), low);
        const __m256i pairs = _mm256_add_epi16(_mm256_maddubs_epi16(hi, qh), _mm256_maddubs_epi16(lo, ql));
        return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    }

    template <int N>
    KNN_SQ_SEARCH_AVX2 static inline void sqSearchNibbleAvx2(const uint8_t* q, const uint8_t* const* d, const int32_t n,
                                                             uint32_t* out) {
        __m256i acc[N];
        for (int k = 0; k < N; ++k) {
            acc[k] = _mm256_setzero_si256();
        }
        int32_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
            const __m256i ql = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + n + i));
            for (int k = 0; k < N; ++k) {
                acc[k] = _mm256_add_epi32(acc[k],
                    sqSearchNibbleBlockAvx2(qh, ql, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d[k] + i))));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = sqSearchHsumEpi32Avx2(acc[k]) + sqSearchNibbleTail(q, d[k], n, i);
        }
    }

    // ---- AVX-512: 64 bytes per step, masked tail (masked-off bytes read as zero and never fault) ----

    static inline __mmask64 sqSearchTailMask(const int32_t remaining) {
        return remaining >= 64 ? ~0ULL : ((1ULL << remaining) - 1);
    }

    KNN_SQ_SEARCH_AVX512BW static inline __m512i sqSearchPopcnt8Avx512(const __m512i v) {
        const __m512i lut = _mm512_set4_epi32(0x04030302, 0x03020201, 0x03020201, 0x02010100);
        const __m512i low = _mm512_set1_epi8(0x0F);
        return _mm512_add_epi8(_mm512_shuffle_epi8(lut, _mm512_and_si512(v, low)),
                               _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), low)));
    }

    KNN_SQ_SEARCH_AVX512BW static inline __m512i sqSearchWeightedPopcntAvx512Bw(const __m512i q0, const __m512i q1,
                                                                               const __m512i q2, const __m512i q3,
                                                                               const __m512i d) {
        __m512i w = sqSearchPopcnt8Avx512(_mm512_and_si512(q0, d));
        w = _mm512_add_epi8(w, _mm512_slli_epi16(sqSearchPopcnt8Avx512(_mm512_and_si512(q1, d)), 1));
        w = _mm512_add_epi8(w, _mm512_slli_epi16(sqSearchPopcnt8Avx512(_mm512_and_si512(q2, d)), 2));
        w = _mm512_add_epi8(w, _mm512_slli_epi16(sqSearchPopcnt8Avx512(_mm512_and_si512(q3, d)), 3));
        return _mm512_sad_epu8(w, _mm512_setzero_si512());
    }

    template <int N>
    KNN_SQ_SEARCH_AVX512BW static inline void sqSearchDibitAvx512Bw(const uint8_t* q, const uint8_t* const* d,
                                                                    const int32_t n, uint32_t* out) {
        __m512i acc0[N], acc1[N];
        for (int k = 0; k < N; ++k) {
            acc0[k] = _mm512_setzero_si512();
            acc1[k] = _mm512_setzero_si512();
        }
        for (int32_t i = 0; i < n; i += 64) {
            const __mmask64 m = sqSearchTailMask(n - i);
            const __m512i q0 = _mm512_maskz_loadu_epi8(m, q + i);
            const __m512i q1 = _mm512_maskz_loadu_epi8(m, q + n + i);
            const __m512i q2 = _mm512_maskz_loadu_epi8(m, q + 2 * static_cast<int64_t>(n) + i);
            const __m512i q3 = _mm512_maskz_loadu_epi8(m, q + 3 * static_cast<int64_t>(n) + i);
            for (int k = 0; k < N; ++k) {
                acc0[k] = _mm512_add_epi64(acc0[k],
                    sqSearchWeightedPopcntAvx512Bw(q0, q1, q2, q3, _mm512_maskz_loadu_epi8(m, d[k] + i)));
                acc1[k] = _mm512_add_epi64(acc1[k],
                    sqSearchWeightedPopcntAvx512Bw(q0, q1, q2, q3, _mm512_maskz_loadu_epi8(m, d[k] + n + i)));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(_mm512_reduce_add_epi64(acc0[k]) + (_mm512_reduce_add_epi64(acc1[k]) << 1));
        }
    }

    // vpopcntq per 64-bit lane: c(q0&d) + 2 c(q1&d) + 4 c(q2&d) + 8 c(q3&d) in u64 lanes.
    KNN_SQ_SEARCH_AVX512POPCNT static inline __m512i sqSearchWeightedPopcntAvx512(const __m512i q0, const __m512i q1,
                                                                                 const __m512i q2, const __m512i q3,
                                                                                 const __m512i d) {
        __m512i w = _mm512_popcnt_epi64(_mm512_and_si512(q0, d));
        w = _mm512_add_epi64(w, _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(q1, d)), 1));
        w = _mm512_add_epi64(w, _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(q2, d)), 2));
        return _mm512_add_epi64(w, _mm512_slli_epi64(_mm512_popcnt_epi64(_mm512_and_si512(q3, d)), 3));
    }

    template <int N>
    KNN_SQ_SEARCH_AVX512POPCNT static inline void sqSearchDibitAvx512Popcnt(const uint8_t* q, const uint8_t* const* d,
                                                                            const int32_t n, uint32_t* out) {
        __m512i acc0[N], acc1[N];
        for (int k = 0; k < N; ++k) {
            acc0[k] = _mm512_setzero_si512();
            acc1[k] = _mm512_setzero_si512();
        }
        for (int32_t i = 0; i < n; i += 64) {
            const __mmask64 m = sqSearchTailMask(n - i);
            const __m512i q0 = _mm512_maskz_loadu_epi8(m, q + i);
            const __m512i q1 = _mm512_maskz_loadu_epi8(m, q + n + i);
            const __m512i q2 = _mm512_maskz_loadu_epi8(m, q + 2 * static_cast<int64_t>(n) + i);
            const __m512i q3 = _mm512_maskz_loadu_epi8(m, q + 3 * static_cast<int64_t>(n) + i);
            for (int k = 0; k < N; ++k) {
                acc0[k] = _mm512_add_epi64(acc0[k],
                    sqSearchWeightedPopcntAvx512(q0, q1, q2, q3, _mm512_maskz_loadu_epi8(m, d[k] + i)));
                acc1[k] = _mm512_add_epi64(acc1[k],
                    sqSearchWeightedPopcntAvx512(q0, q1, q2, q3, _mm512_maskz_loadu_epi8(m, d[k] + n + i)));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(_mm512_reduce_add_epi64(acc0[k]) + (_mm512_reduce_add_epi64(acc1[k]) << 1));
        }
    }

    KNN_SQ_SEARCH_AVX512BW static inline __m512i sqSearchNibbleBlockAvx512Bw(const __m512i qh, const __m512i ql,
                                                                            const __m512i p) {
        const __m512i low = _mm512_set1_epi8(0x0F);
        const __m512i lo = _mm512_and_si512(p, low);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(p, 4), low);
        const __m512i pairs = _mm512_add_epi16(_mm512_maddubs_epi16(hi, qh), _mm512_maddubs_epi16(lo, ql));
        return _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
    }

    template <int N>
    KNN_SQ_SEARCH_AVX512BW static inline void sqSearchNibbleAvx512Bw(const uint8_t* q, const uint8_t* const* d,
                                                                     const int32_t n, uint32_t* out) {
        __m512i acc[N];
        for (int k = 0; k < N; ++k) {
            acc[k] = _mm512_setzero_si512();
        }
        for (int32_t i = 0; i < n; i += 64) {
            const __mmask64 m = sqSearchTailMask(n - i);
            const __m512i qh = _mm512_maskz_loadu_epi8(m, q + i);
            const __m512i ql = _mm512_maskz_loadu_epi8(m, q + n + i);
            for (int k = 0; k < N; ++k) {
                acc[k] = _mm512_add_epi32(acc[k], sqSearchNibbleBlockAvx512Bw(qh, ql, _mm512_maskz_loadu_epi8(m, d[k] + i)));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc[k]));
        }
    }

    // vpdpbusd: u8 (document nibble) x s8 (query code, 0..15) summed in groups of four into int32 lanes.
    template <int N>
    KNN_SQ_SEARCH_AVX512VNNI static inline void sqSearchNibbleAvx512Vnni(const uint8_t* q, const uint8_t* const* d,
                                                                         const int32_t n, uint32_t* out) {
        const __m512i low = _mm512_set1_epi8(0x0F);
        __m512i acc[N];
        for (int k = 0; k < N; ++k) {
            acc[k] = _mm512_setzero_si512();
        }
        for (int32_t i = 0; i < n; i += 64) {
            const __mmask64 m = sqSearchTailMask(n - i);
            const __m512i qh = _mm512_maskz_loadu_epi8(m, q + i);
            const __m512i ql = _mm512_maskz_loadu_epi8(m, q + n + i);
            for (int k = 0; k < N; ++k) {
                const __m512i p = _mm512_maskz_loadu_epi8(m, d[k] + i);
                acc[k] = _mm512_dpbusd_epi32(acc[k], _mm512_and_si512(_mm512_srli_epi16(p, 4), low), qh);
                acc[k] = _mm512_dpbusd_epi32(acc[k], _mm512_and_si512(p, low), ql);
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc[k]));
        }
    }

    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX2, sqSearchDibitAvx2)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX2, sqSearchNibbleAvx2)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX512BW, sqSearchDibitAvx512Bw)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX512BW, sqSearchNibbleAvx512Bw)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX512POPCNT, sqSearchDibitAvx512Popcnt)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_AVX512VNNI, sqSearchNibbleAvx512Vnni)

#undef KNN_SQ_SEARCH_AVX2
#undef KNN_SQ_SEARCH_AVX512BW
#undef KNN_SQ_SEARCH_AVX512POPCNT
#undef KNN_SQ_SEARCH_AVX512VNNI
#endif  // KNN_SQ_SEARCH_X86



#if defined(KNN_SQ_SEARCH_ARM)
    // ---- NEON (baseline on aarch64): 16 bytes per step, scalar tail ----

    static inline uint8x16_t sqSearchWeightedVcnt(const uint8x16_t q0, const uint8x16_t q1, const uint8x16_t q2,
                                                  const uint8x16_t q3, const uint8x16_t d) {
        uint8x16_t w = vcntq_u8(vandq_u8(q0, d));
        w = vaddq_u8(w, vshlq_n_u8(vcntq_u8(vandq_u8(q1, d)), 1));
        w = vaddq_u8(w, vshlq_n_u8(vcntq_u8(vandq_u8(q2, d)), 2));
        return vaddq_u8(w, vshlq_n_u8(vcntq_u8(vandq_u8(q3, d)), 3));  // <= 120 per byte
    }

    static inline uint64_t sqSearchDibitTailArm(const uint8_t* q, const uint8_t* d, const int32_t n, const int32_t from) {
        uint64_t acc = 0;
        for (int32_t i = from; i < n; ++i) {
            for (int32_t p = 0; p < 4; ++p) {
                const uint8_t qb = q[static_cast<int64_t>(p) * n + i];
                acc += static_cast<uint64_t>(__builtin_popcount(qb & d[i]) + 2 * __builtin_popcount(qb & d[n + i])) << p;
            }
        }
        return acc;
    }

    static inline uint32_t sqSearchNibbleTailArm(const uint8_t* q, const uint8_t* d, const int32_t n, const int32_t from) {
        uint32_t sum = 0;
        for (int32_t i = from; i < n; ++i) {
            sum += static_cast<uint32_t>(q[i]) * static_cast<uint32_t>(d[i] >> 4)
                 + static_cast<uint32_t>(q[n + i]) * static_cast<uint32_t>(d[i] & 0x0F);
        }
        return sum;
    }

    template <int N>
    static inline void sqSearchDibitNeon(const uint8_t* q, const uint8_t* const* d, const int32_t n, uint32_t* out) {
        uint32x4_t acc0[N], acc1[N];
        for (int k = 0; k < N; ++k) {
            acc0[k] = vdupq_n_u32(0);
            acc1[k] = vdupq_n_u32(0);
        }
        int32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t q0 = vld1q_u8(q + i);
            const uint8x16_t q1 = vld1q_u8(q + n + i);
            const uint8x16_t q2 = vld1q_u8(q + 2 * static_cast<int64_t>(n) + i);
            const uint8x16_t q3 = vld1q_u8(q + 3 * static_cast<int64_t>(n) + i);
            for (int k = 0; k < N; ++k) {
                acc0[k] = vpadalq_u16(acc0[k], vpaddlq_u8(sqSearchWeightedVcnt(q0, q1, q2, q3, vld1q_u8(d[k] + i))));
                acc1[k] = vpadalq_u16(acc1[k], vpaddlq_u8(sqSearchWeightedVcnt(q0, q1, q2, q3, vld1q_u8(d[k] + n + i))));
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = static_cast<uint32_t>(static_cast<uint64_t>(vaddvq_u32(acc0[k]))
                                           + (static_cast<uint64_t>(vaddvq_u32(acc1[k])) << 1)
                                           + sqSearchDibitTailArm(q, d[k], n, i));
        }
    }

    // umull of the 8-byte halves into u16 (<= 225), pairwise-accumulated into u32 lanes.
    template <int N>
    static inline void sqSearchNibbleNeon(const uint8_t* q, const uint8_t* const* d, const int32_t n, uint32_t* out) {
        const uint8x16_t low = vdupq_n_u8(0x0F);
        uint32x4_t acc[N];
        for (int k = 0; k < N; ++k) {
            acc[k] = vdupq_n_u32(0);
        }
        int32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t qh = vld1q_u8(q + i);
            const uint8x16_t ql = vld1q_u8(q + n + i);
            for (int k = 0; k < N; ++k) {
                const uint8x16_t p = vld1q_u8(d[k] + i);
                const uint8x16_t hi = vshrq_n_u8(p, 4);
                const uint8x16_t lo = vandq_u8(p, low);
                uint16x8_t s = vmull_u8(vget_low_u8(hi), vget_low_u8(qh));
                s = vmlal_u8(s, vget_low_u8(lo), vget_low_u8(ql));        // <= 450 per lane
                uint16x8_t t = vmull_u8(vget_high_u8(hi), vget_high_u8(qh));
                t = vmlal_u8(t, vget_high_u8(lo), vget_high_u8(ql));
                acc[k] = vpadalq_u16(vpadalq_u16(acc[k], s), t);
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = vaddvq_u32(acc[k]) + sqSearchNibbleTailArm(q, d[k], n, i);
        }
    }

#if defined(__linux__)
#define KNN_SQ_SEARCH_DOTPROD __attribute__((target("arch=armv8.2-a+dotprod")))

    // udot: four u8 x u8 products summed into each u32 lane.
    template <int N>
    KNN_SQ_SEARCH_DOTPROD static inline void sqSearchNibbleNeonDot(const uint8_t* q, const uint8_t* const* d,
                                                                   const int32_t n, uint32_t* out) {
        const uint8x16_t low = vdupq_n_u8(0x0F);
        uint32x4_t acc[N];
        for (int k = 0; k < N; ++k) {
            acc[k] = vdupq_n_u32(0);
        }
        int32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t qh = vld1q_u8(q + i);
            const uint8x16_t ql = vld1q_u8(q + n + i);
            for (int k = 0; k < N; ++k) {
                const uint8x16_t p = vld1q_u8(d[k] + i);
                acc[k] = vdotq_u32(acc[k], vshrq_n_u8(p, 4), qh);
                acc[k] = vdotq_u32(acc[k], vandq_u8(p, low), ql);
            }
        }
        for (int k = 0; k < N; ++k) {
            out[k] = vaddvq_u32(acc[k]) + sqSearchNibbleTailArm(q, d[k], n, i);
        }
    }
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_DOTPROD, sqSearchNibbleNeonDot)
    KNN_SQ_SEARCH_DOTPROD static void sqSearchNibbleNeonDotEight(const uint8_t* q, const uint8_t* const* d, const int32_t n,
                                                                 uint32_t* out) {
        sqSearchNibbleNeonDot<8>(q, d, n, out);
    }
#undef KNN_SQ_SEARCH_DOTPROD
#endif  // __linux__
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_NO_TARGET, sqSearchDibitNeon)
    KNN_SQ_SEARCH_ENTRY_POINTS(KNN_SQ_SEARCH_NO_TARGET, sqSearchNibbleNeon)
#endif  // KNN_SQ_SEARCH_ARM
#undef KNN_SQ_SEARCH_ENTRY_POINTS
#undef KNN_SQ_SEARCH_NO_TARGET

    // ---- Runtime selection ----

    struct SqSearchCpuFeatures {
        bool avx2 = false;
        bool avx512bw = false;
        bool avx512vpopcntdq = false;
        bool avx512vnni = false;
        bool neonDotProd = false;
    };

    static inline const SqSearchCpuFeatures& sqSearchCpuFeatures() {
        static const SqSearchCpuFeatures features = [] {
            SqSearchCpuFeatures f;
#if defined(KNN_SQ_SEARCH_X86)
            __builtin_cpu_init();
            // libgcc also checks via XGETBV that the OS saves the YMM / ZMM / opmask state for these.
            f.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
            f.avx512bw = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
            f.avx512vpopcntdq = f.avx512bw && __builtin_cpu_supports("avx512vpopcntdq");
            f.avx512vnni = f.avx512bw && __builtin_cpu_supports("avx512vnni");
#elif defined(KNN_SQ_SEARCH_ARM) && defined(__linux__)
            f.neonDotProd = (getauxval(AT_HWCAP) & KNN_SQ_HWCAP_ASIMDDP) != 0;
#endif
            return f;
        }();
        return features;
    }

    // Resolves the kernel for document code width `bits` (2 or 4). `Auto` picks the fastest implementation the CPU
    // supports and always succeeds. Any other request returns false, leaving `out` untouched, when that implementation does
    // not exist for this width / architecture or the CPU lacks the instructions (used by tests to run every kernel).
    static inline bool resolveSqSearchKernel(const int32_t bits, SqSearchImpl requested, SqSearchKernel& out) {
        if (bits != 2 && bits != 4) {
            return false;
        }
        const SqSearchCpuFeatures& cpu = sqSearchCpuFeatures();
        (void) cpu;
        if (requested == SqSearchImpl::Auto) {
#if defined(KNN_SQ_SEARCH_X86)
            if (bits == 2) {
                requested = cpu.avx512vpopcntdq ? SqSearchImpl::X86Avx512Popcnt
                          : cpu.avx512bw        ? SqSearchImpl::X86Avx512Bw
                          : cpu.avx2            ? SqSearchImpl::X86Avx2
                                                : SqSearchImpl::Scalar;
            } else {
                requested = cpu.avx512vnni ? SqSearchImpl::X86Avx512Vnni
                          : cpu.avx512bw   ? SqSearchImpl::X86Avx512Bw
                          : cpu.avx2       ? SqSearchImpl::X86Avx2
                                           : SqSearchImpl::Scalar;
            }
#elif defined(KNN_SQ_SEARCH_ARM)
            requested = (bits == 4 && cpu.neonDotProd) ? SqSearchImpl::NeonDotProd : SqSearchImpl::Neon;
#else
            requested = SqSearchImpl::Scalar;
#endif
        }
        if (requested == SqSearchImpl::Scalar) {
            out = bits == 2 ? SqSearchKernel{requested, &sqSearchDibitScalar, &sqSearchDibit4Scalar}
                            : SqSearchKernel{requested, &sqSearchNibbleScalar, &sqSearchNibble4Scalar};
            return true;
        }
#if defined(KNN_SQ_SEARCH_X86)
        if (requested == SqSearchImpl::X86Avx2 && cpu.avx2) {
            out = bits == 2 ? SqSearchKernel{requested, &sqSearchDibitAvx2One, &sqSearchDibitAvx2Four}
                            : SqSearchKernel{requested, &sqSearchNibbleAvx2One, &sqSearchNibbleAvx2Four};
            return true;
        }
        if (requested == SqSearchImpl::X86Avx512Bw && cpu.avx512bw) {
            out = bits == 2 ? SqSearchKernel{requested, &sqSearchDibitAvx512BwOne, &sqSearchDibitAvx512BwFour}
                            : SqSearchKernel{requested, &sqSearchNibbleAvx512BwOne, &sqSearchNibbleAvx512BwFour};
            return true;
        }
        if (requested == SqSearchImpl::X86Avx512Popcnt && bits == 2 && cpu.avx512vpopcntdq) {
            out = {requested, &sqSearchDibitAvx512PopcntOne, &sqSearchDibitAvx512PopcntFour};
            return true;
        }
        if (requested == SqSearchImpl::X86Avx512Vnni && bits == 4 && cpu.avx512vnni) {
            out = {requested, &sqSearchNibbleAvx512VnniOne, &sqSearchNibbleAvx512VnniFour};
            return true;
        }
#endif
#if defined(KNN_SQ_SEARCH_ARM)
        if (requested == SqSearchImpl::Neon) {
            out = bits == 2 ? SqSearchKernel{requested, &sqSearchDibitNeonOne, &sqSearchDibitNeonFour}
                            : SqSearchKernel{requested, &sqSearchNibbleNeonOne, &sqSearchNibbleNeonFour};
            return true;
        }
#if defined(__linux__)
        if (requested == SqSearchImpl::NeonDotProd && bits == 4 && cpu.neonDotProd) {
            out = {requested, &sqSearchNibbleNeonDotOne, &sqSearchNibbleNeonDotFour, &sqSearchNibbleNeonDotEight};
            return true;
        }
#endif
#endif
        return false;
    }

    // The auto-selected kernel for `bits` (2 or 4), resolved once per process.
    static inline const SqSearchKernel& sqSearchKernel(const int32_t bits) {
        static const SqSearchKernel two = [] { SqSearchKernel k; resolveSqSearchKernel(2, SqSearchImpl::Auto, k); return k; }();
        static const SqSearchKernel four = [] { SqSearchKernel k; resolveSqSearchKernel(4, SqSearchImpl::Auto, k); return k; }();
        return bits == 2 ? two : four;
    }

    // Scores `internalVectorIds[0..numVectors)` of a sq_format.h format with `kernel`: eight records per kernel call when the
    // kernel has an eight-record entry point, then four, then one at a time. `dotBytes` is the stripe length for 2-bit codes and the packed length for 4-bit codes; the format
    // locates the records and their corrections, and `epilogue` turns a dot product and those corrections into a score.
    template <typename Format, typename Epilogue>
    static inline void scoreLuceneSqMultiBit(const Format& format, const Epilogue& epilogue, const SqSearchKernel& kernel,
                                             const uint8_t* queryPtr, const int32_t dotBytes, int32_t* internalVectorIds,
                                             float* scores, const int32_t numVectors) {
        int32_t i = 0;
        uint8_t* vectors[8];
        uint32_t dots[8];
        if (kernel.dp8 != nullptr) {
            for (; i + 8 <= numVectors; i += 8) {
                format.getVectorPointersInBulk(vectors, &internalVectorIds[i], 8);
                kernel.dp8(queryPtr, vectors, dotBytes, dots);
                for (int32_t k = 0; k < 8; ++k) {
                    float ax, lx, additional, x1;
                    format.readDataCorrections(vectors[k], internalVectorIds[i + k], ax, lx, additional, x1);
                    scores[i + k] = epilogue.finish(ax, lx, additional, x1, static_cast<float>(dots[k]));
                }
            }
        }
        for (; i + 4 <= numVectors; i += 4) {
            format.getVectorPointersInBulk(vectors, &internalVectorIds[i], 4);
            kernel.dp4(queryPtr, vectors[0], vectors[1], vectors[2], vectors[3], dotBytes, dots);
            for (int32_t k = 0; k < 4; ++k) {
                float ax, lx, additional, x1;
                format.readDataCorrections(vectors[k], internalVectorIds[i + k], ax, lx, additional, x1);
                scores[i + k] = epilogue.finish(ax, lx, additional, x1, static_cast<float>(dots[k]));
            }
        }
        for (; i < numVectors; ++i) {
            const uint8_t* vector = format.getVectorPointer(internalVectorIds[i]);
            const uint32_t dot = kernel.dp(queryPtr, vector, dotBytes);
            float ax, lx, additional, x1;
            format.readDataCorrections(vector, internalVectorIds[i], ax, lx, additional, x1);
            scores[i] = epilogue.finish(ax, lx, additional, x1, static_cast<float>(dot));
        }
    }
}

#endif  // OPENSEARCH_KNN_SIMD_SQ_SEARCH_MULTIBIT_KERNELS_H
