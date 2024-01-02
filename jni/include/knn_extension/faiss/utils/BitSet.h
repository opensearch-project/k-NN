/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/impl/platform_macros.h>
#include <limits>

using idx_t = faiss::idx_t;

struct BitSet {
    const int NO_MORE_DOCS = std::numeric_limits<int>::max();
    /**
     * Returns the index of the first set bit starting at the index specified.
     * NO_MORE_DOCS is returned if there are no more set bits.
     */
    virtual idx_t next_set_bit(idx_t index) const = 0;
    virtual ~BitSet() = default;
};


/**
 * BitSet of fixed length (numBits), implemented using an array of unit64.
 * See https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/FixedBitSet.java
 *
 * Here a block is 64 bit. However, for simplicity let's assume its size is 8 bits.
 * Then, if have an array of 3, 7, and 10, it will be represented in bitmap as follow.
 *            [0]      [1]
 * bitmap: 10001000 00000100
 *
 * for next_set_bit call with 4
 * 1. it looks for words[0]
 * 2. words[0] >> 4
 * 3. count trailing zero of the result from step 2 which is 3
 * 4. return 4(current index) + 3(result from step 3)
 */
struct FixedBitSet : public BitSet {
    // The number of bits in use
    idx_t num_bits;

    // The exact number of longs needed to hold num_bits
    size_t num_words;

    // Array of uint64_t holding the bits
    // Using uint64_t to leverage function __builtin_ctzll which is defined in faiss/impl/platform_macros.h
    uint64_t* words;

    FixedBitSet(const int* int_array, const int length);
    idx_t next_set_bit(idx_t index) const;
    ~FixedBitSet();
};
