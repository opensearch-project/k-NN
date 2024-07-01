// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_methods.h"
#include "faiss/index_factory.h"

namespace knn_jni {
namespace faiss_wrapper {

faiss::Index* FaissMethods::indexFactory(int d, const char* description, faiss::MetricType metric) {
    return faiss::index_factory(d, description, metric);
}

faiss::IndexBinary* FaissMethods::indexBinaryFactory(int d, const char* description) {
    return faiss::index_binary_factory(d, description);
}

faiss::IndexIDMapTemplate<faiss::Index>* FaissMethods::indexIdMap(faiss::Index* index) {
    return new faiss::IndexIDMap(index);
}

faiss::IndexIDMapTemplate<faiss::IndexBinary>* FaissMethods::indexBinaryIdMap(faiss::IndexBinary* index) {
    return new faiss::IndexBinaryIDMap(index);
}

void FaissMethods::writeIndex(const faiss::Index* idx, const char* fname) {
    faiss::write_index(idx, fname);
}
void FaissMethods::writeIndexBinary(const faiss::IndexBinary* idx, const char* fname) {
    faiss::write_index_binary(idx, fname);
}

} // namespace faiss_wrapper
} // namesapce knn_jni
