// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_util.h"
#include <algorithm>

std::unique_ptr<faiss::IDGrouperBitmap> faiss_util::buildIDGrouperBitmap(int *parentIdsArray,  int parentIdsLength, std::vector<uint64_t>* bitmap) {
    const int* maxValue = std::max_element(parentIdsArray, parentIdsArray + parentIdsLength);
    int num_bits = *maxValue + 1;
    int num_blocks = (num_bits >> 6) + 1; // div by 64
    bitmap->resize(num_blocks, 0);
    std::unique_ptr<faiss::IDGrouperBitmap> idGrouper(new faiss::IDGrouperBitmap(num_blocks, bitmap->data()));
    for (int i = 0; i < parentIdsLength; i++) {
        idGrouper->set_group(parentIdsArray[i]);
    }
    return idGrouper;
}
