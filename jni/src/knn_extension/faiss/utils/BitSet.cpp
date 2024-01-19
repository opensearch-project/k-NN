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
    this->num_bits = *maxValue + 1;
    this->num_words = (num_bits >> 6) + 1; // div by 64
    this->words = new uint64_t[this->num_words]();
    for(int i = 0 ; i < length ; i ++) {
        int value = int_array[i];
        int bitset_array_index = value >> 6;
        this->words[bitset_array_index] |= 1ULL << (value & 63); // Equivalent of 1ULL << (value % 64)
    }
}

idx_t FixedBitSet::next_set_bit(idx_t index) const {
    assert(index >= 0 && "index shouldn't be less than zero");
    assert(index < this->num_bits && "index should be less than total number of bits");

    idx_t i = index >> 6; // div by 64
    uint64_t word = this->words[i] >> (index & 63); // Equivalent of words[i] >> (index % 64)
    // word is non zero after right shift, it means, next set bit is in current word
    // The index of set bit is "given index" + "trailing zero in the right shifted word"
    if (word != 0) {
      return index + __builtin_ctzll(word);
    }

    while (++i < this->num_words) {
      word = this->words[i];
      if (word != 0) {
        return (i << 6) + __builtin_ctzll(word);
      }
    }

    return NO_MORE_DOCS;
}

FixedBitSet::~FixedBitSet() {
    delete this->words;
}
