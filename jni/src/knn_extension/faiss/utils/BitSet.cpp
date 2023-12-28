/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <algorithm>
#include "BitSet.h"

FixedBitSet::FixedBitSet(const int* int_array, const int length){
    assert(int_array && "int_array should not be null");
    const int* maxValue = std::max_element(int_array, int_array + length);
    this->numBits = (*maxValue >> 6) + 1; // div by 64
    this->bitmap = new uint64_t[this->numBits]();
    for(int i = 0 ; i < length ; i ++) {
        int value = int_array[i];
        int bitsetArrayIndex = value >> 6;
        this->bitmap[bitsetArrayIndex] |= 1ULL << (value & 63); // Equivalent of 1ULL << (value % 64)
    }
}

idx_t FixedBitSet::next_set_bit(idx_t index) const {
    idx_t i = index >> 6; // div by 64
    uint64_t word = this->bitmap[i] >> (index & 63); // Equivalent of bitmap[i] >> (index % 64)

    if (word != 0) {
      return index + __builtin_ctzll(word);
    }

    while (++i < this->numBits) {
      word = this->bitmap[i];
      if (word != 0) {
        return (i << 6) + __builtin_ctzll(word);
      }
    }

    return NO_MORE_DOCS;
}

FixedBitSet::~FixedBitSet() {
    delete this->bitmap;
}
