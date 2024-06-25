// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#ifndef OPENSEARCH_KNN_FAISS_METHODS_H
#define OPENSEARCH_KNN_FAISS_METHODS_H

#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"

namespace knn_jni {
namespace faiss_wrapper {

/**
 * A class having wrapped faiss methods
 *
 * This class helps to mock faiss methods during unit test
 */
class FaissMethods {
public:
    FaissMethods() = default;
    virtual faiss::Index* indexFactory(int d, const char* description, faiss::MetricType metric);
    virtual faiss::IndexBinary* indexBinaryFactory(int d, const char* description);
    virtual faiss::IndexIDMapTemplate<faiss::Index>* indexIdMap(faiss::Index* index);
    virtual faiss::IndexIDMapTemplate<faiss::IndexBinary>* indexBinaryIdMap(faiss::IndexBinary* index);
    virtual void writeIndex(const faiss::Index* idx, const char* fname);
    virtual void writeIndexBinary(const faiss::IndexBinary* idx, const char* fname);
    virtual ~FaissMethods() = default;
};

} //namespace faiss_wrapper
} //namespace knn_jni


#endif //OPENSEARCH_KNN_FAISS_METHODS_H