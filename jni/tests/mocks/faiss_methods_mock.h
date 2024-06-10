/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

 #ifndef OPENSEARCH_KNN_FAISS_METHODS_MOCK_H
 #define OPENSEARCH_KNN_FAISS_METHODS_MOCK_H

#include "faiss_methods.h"
#include <gmock/gmock.h>

class MockFaissMethods : public knn_jni::faiss_wrapper::FaissMethods {
public:
    MOCK_METHOD(faiss::Index*, indexFactory, (int d, const char* description, faiss::MetricType metric), (override));
    MOCK_METHOD(faiss::IndexBinary*, indexBinaryFactory, (int d, const char* description), (override));
    MOCK_METHOD(faiss::IndexIDMapTemplate<faiss::Index>*, indexIdMap, (faiss::Index* index), (override));
    MOCK_METHOD(faiss::IndexIDMapTemplate<faiss::IndexBinary>*, indexBinaryIdMap, (faiss::IndexBinary* index), (override));
    MOCK_METHOD(void, writeIndex, (const faiss::Index* idx, const char* fname), (override));
    MOCK_METHOD(void, writeIndexBinary, (const faiss::IndexBinary* idx, const char* fname), (override));
};

#endif  // OPENSEARCH_KNN_FAISS_METHODS_MOCK_H