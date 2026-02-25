// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

/**
 * This file contains util methods which are free of JNI to be used in faiss_wrapper.cpp
 */

#ifndef OPENSEARCH_KNN_FAISS_UTIL_H
#define OPENSEARCH_KNN_FAISS_UTIL_H

#include "faiss/impl/IDGrouper.h"
#include <memory>

namespace faiss_util {
    std::unique_ptr<faiss::IDGrouperBitmap> buildIDGrouperBitmap(int *parentIdsArray,  int parentIdsLength, std::vector<uint64_t>* bitmap);
};


#endif //OPENSEARCH_KNN_FAISS_UTIL_H
