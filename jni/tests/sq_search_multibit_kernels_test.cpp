/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simd/similarity_function/sq_search_multibit_kernels.h"

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define KNN_SQ_SEARCH_TEST_HAS_MMAP 1
#endif

#include "gtest/gtest.h"

using namespace knn_jni::simd::similarity_function;

namespace {

const SqSearchImpl kAllImpls[] = {
    SqSearchImpl::Scalar, SqSearchImpl::X86Avx2, SqSearchImpl::X86Avx512Bw, SqSearchImpl::X86Avx512Popcnt,
    SqSearchImpl::X86Avx512Vnni, SqSearchImpl::Neon, SqSearchImpl::NeonDotProd,
};

// Independent reference: Lucene's int4DibitDotProduct / int4DotProductSinglePacked, bit by bit, in 64-bit.
uint64_t referenceDot(const int32_t bits, const uint8_t* q, const uint8_t* d, const int32_t n) {
    uint64_t dot = 0;
    if (bits == 2) {
        for (int32_t plane = 0; plane < 2; ++plane) {
            for (int32_t p = 0; p < 4; ++p) {
                for (int32_t i = 0; i < n; ++i) {
                    for (int32_t bit = 0; bit < 8; ++bit) {
                        const uint64_t qb = (q[p * n + i] >> bit) & 1, db = (d[plane * n + i] >> bit) & 1;
                        dot += (qb & db) << (p + plane);
                    }
                }
            }
        }
        return dot;
    }
    for (int32_t i = 0; i < n; ++i) {
        dot += uint64_t(q[i]) * (d[i] >> 4) + uint64_t(q[n + i]) * (d[i] & 0x0F);
    }
    return dot;
}

// Query and document byte lengths for `n`: 2-bit 4n / 2n, 4-bit 2n / n.
int32_t queryBytes(const int32_t bits, const int32_t n) { return bits == 2 ? 4 * n : 2 * n; }
int32_t docBytes(const int32_t bits, const int32_t n) { return bits == 2 ? 2 * n : n; }

// Fills like a real code: 4-bit query codes are 0..15 (one per byte); everything else is random bits, or all set.
void fill(std::vector<uint8_t>& v, const bool isFourBitQuery, const bool allSet, std::mt19937& rng) {
    for (auto& b : v) {
        const uint8_t r = allSet ? 0xFF : static_cast<uint8_t>(rng());
        b = isFourBitQuery ? static_cast<uint8_t>(r & 0x0F) : r;
    }
}

// Runs dp and dp4 of every implementation the CPU has on (q, d0..d3) and checks them against the reference.
void checkAllImpls(const int32_t bits, const int32_t n, const uint8_t* q, const uint8_t* const* d, const std::string& ctx) {
    uint64_t expected[4];
    for (int k = 0; k < 4; ++k) {
        expected[k] = referenceDot(bits, q, d[k], n);
    }
    for (SqSearchImpl impl : kAllImpls) {
        SqSearchKernel kernel;
        if (!resolveSqSearchKernel(bits, impl, kernel)) {
            continue;
        }
        const std::string where = ctx + " impl=" + sqSearchImplName(kernel.impl) + " bits=" + std::to_string(bits)
            + " n=" + std::to_string(n);
        ASSERT_EQ(uint64_t(kernel.dp(q, d[0], n)), expected[0]) << where;
        uint32_t out[4];
        kernel.dp4(q, d[0], d[1], d[2], d[3], n, out);
        for (int k = 0; k < 4; ++k) {
            ASSERT_EQ(uint64_t(out[k]), expected[k]) << where << " k=" << k;
        }
        if (kernel.dp8 != nullptr) {
            const uint8_t* d8[8] = {d[0], d[1], d[2], d[3], d[3], d[2], d[1], d[0]};
            uint32_t out8[8];
            kernel.dp8(q, d8, n, out8);
            for (int k = 0; k < 8; ++k) {
                ASSERT_EQ(uint64_t(out8[k]), expected[k < 4 ? k : 7 - k]) << where << " dp8 k=" << k;
            }
        }
    }
}

class SqSearchKernelTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(SqSearchKernelTest, EveryImplementationMatchesReference) {
    const int32_t bits = GetParam();
    std::mt19937 rng(7);
    std::vector<int32_t> sizes;
    for (int32_t n = 1; n <= 300; ++n) {
        sizes.push_back(n);
    }
    for (int32_t n : {511, 512, 513, 1000, 2000, 8000}) {
        sizes.push_back(n);
    }
    for (int32_t n : sizes) {
        for (bool allSet : {false, true}) {
            std::vector<uint8_t> q(queryBytes(bits, n));
            fill(q, bits == 4, allSet, rng);
            std::vector<std::vector<uint8_t>> docs(4, std::vector<uint8_t>(docBytes(bits, n)));
            for (auto& doc : docs) {
                fill(doc, false, allSet, rng);
            }
            const uint8_t* d[4] = {docs[0].data(), docs[1].data(), docs[2].data(), docs[3].data()};
            checkAllImpls(bits, n, q.data(), d, allSet ? "allSet" : "random");
            if (HasFatalFailure()) {
                return;
            }
        }
    }
}

TEST_P(SqSearchKernelTest, AutoSelectsFastestAvailable) {
    const int32_t bits = GetParam();
    SqSearchKernel kernel;
    ASSERT_TRUE(resolveSqSearchKernel(bits, SqSearchImpl::Auto, kernel));
    EXPECT_NE(kernel.impl, SqSearchImpl::Auto);
    EXPECT_EQ(sqSearchKernel(bits).impl, kernel.impl);
    const SqSearchCpuFeatures& cpu = sqSearchCpuFeatures();
    (void) cpu;
    SqSearchImpl expected = SqSearchImpl::Scalar;
#if defined(KNN_SQ_SEARCH_X86)
    if (bits == 2) {
        expected = cpu.avx512vpopcntdq ? SqSearchImpl::X86Avx512Popcnt
                 : cpu.avx512bw        ? SqSearchImpl::X86Avx512Bw
                 : cpu.avx2            ? SqSearchImpl::X86Avx2
                                       : SqSearchImpl::Scalar;
    } else {
        expected = cpu.avx512vnni ? SqSearchImpl::X86Avx512Vnni
                 : cpu.avx512bw   ? SqSearchImpl::X86Avx512Bw
                 : cpu.avx2       ? SqSearchImpl::X86Avx2
                                  : SqSearchImpl::Scalar;
    }
#elif defined(KNN_SQ_SEARCH_ARM)
    expected = (bits == 4 && cpu.neonDotProd) ? SqSearchImpl::NeonDotProd : SqSearchImpl::Neon;
#endif
    EXPECT_EQ(kernel.impl, expected) << "auto picked " << sqSearchImplName(kernel.impl);
}

TEST_P(SqSearchKernelTest, RejectsImplementationsForOtherWidths) {
    const int32_t bits = GetParam();
    SqSearchKernel kernel;
    // VPOPCNTDQ is a 2-bit-only kernel; VNNI and NEON dotprod are 4-bit-only kernels.
    for (SqSearchImpl impl : bits == 2 ? std::vector<SqSearchImpl>{SqSearchImpl::X86Avx512Vnni, SqSearchImpl::NeonDotProd}
                                       : std::vector<SqSearchImpl>{SqSearchImpl::X86Avx512Popcnt}) {
        EXPECT_FALSE(resolveSqSearchKernel(bits, impl, kernel)) << sqSearchImplName(impl);
    }
    EXPECT_FALSE(resolveSqSearchKernel(1, SqSearchImpl::Scalar, kernel));
    EXPECT_FALSE(resolveSqSearchKernel(8, SqSearchImpl::Auto, kernel));
    EXPECT_TRUE(resolveSqSearchKernel(bits, SqSearchImpl::Scalar, kernel));
}

#if defined(KNN_SQ_SEARCH_TEST_HAS_MMAP)
// len writable bytes flush against a PROT_NONE page, right before it (guardAfter) or right after it.
class GuardedBuffer {
public:
    GuardedBuffer(const size_t len, const bool guardAfter) {
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const size_t dataPages = (len + page - 1) / page;
        mapLen_ = (dataPages + 2) * page;
        void* m = mmap(nullptr, mapLen_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        map_ = static_cast<uint8_t*>(m);
        if (mprotect(map_, page, PROT_NONE) != 0 || mprotect(map_ + (dataPages + 1) * page, page, PROT_NONE) != 0) {
            munmap(map_, mapLen_);
            throw std::runtime_error("mprotect failed");
        }
        data_ = guardAfter ? map_ + (dataPages + 1) * page - len : map_ + page;
    }
    ~GuardedBuffer() { munmap(map_, mapLen_); }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;
    uint8_t* data() { return data_; }

private:
    uint8_t* map_ = nullptr;
    uint8_t* data_ = nullptr;
    size_t mapLen_ = 0;
};

TEST_P(SqSearchKernelTest, KernelsNeverReadOutsideTheCodes) {
    // Query and documents flush against a guard page on either side: a tail that touches one byte outside them faults.
    // n = 1..130 covers every remainder of the 16/32/64-byte steps after zero, one and two full steps.
    const int32_t bits = GetParam();
    std::mt19937 rng(11);
    std::vector<int32_t> sizes;
    for (int32_t n = 1; n <= 130; ++n) {
        sizes.push_back(n);
    }
    for (int32_t n : {255, 256, 257, 4095, 4096, 4097}) {
        sizes.push_back(n);
    }
    for (int32_t n : sizes) {
        for (bool guardAfter : {true, false}) {
            GuardedBuffer q(queryBytes(bits, n), guardAfter);
            std::unique_ptr<GuardedBuffer> docs[4];
            const uint8_t* d[4];
            for (int k = 0; k < 4; ++k) {
                docs[k].reset(new GuardedBuffer(docBytes(bits, n), guardAfter));
                d[k] = docs[k]->data();
            }
            for (bool allSet : {false, true}) {
                for (int32_t i = 0; i < queryBytes(bits, n); ++i) {
                    const uint8_t r = allSet ? 0xFF : static_cast<uint8_t>(rng());
                    q.data()[i] = bits == 4 ? static_cast<uint8_t>(r & 0x0F) : r;
                }
                for (int k = 0; k < 4; ++k) {
                    for (int32_t i = 0; i < docBytes(bits, n); ++i) {
                        docs[k]->data()[i] = allSet ? 0xFF : static_cast<uint8_t>(rng());
                    }
                }
                checkAllImpls(bits, n, q.data(), d, std::string(guardAfter ? "guard after" : "guard before"));
                if (HasFatalFailure()) {
                    return;
                }
            }
        }
    }
}
#endif  // KNN_SQ_SEARCH_TEST_HAS_MMAP

INSTANTIATE_TEST_SUITE_P(Bits, SqSearchKernelTest, ::testing::Values(2, 4),
                         [](const ::testing::TestParamInfo<int32_t>& info) { return "B" + std::to_string(info.param); });

}  // namespace
