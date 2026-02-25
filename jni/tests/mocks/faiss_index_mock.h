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

 #ifndef OPENSEARCH_KNN_FAISS_INDEX_MOCK_H
 #define OPENSEARCH_KNN_FAISS_INDEX_MOCK_H

#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include <gmock/gmock.h>

using idx_t = int64_t;

class MockIndex : public faiss::Index {
public:
    MOCK_METHOD(void, add, (idx_t n, const float* x), (override));
    MOCK_METHOD(void, search, (idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const faiss::SearchParameters* params), (const, override));
    MOCK_METHOD(void, reset, (), (override));
};

class MockIndexBinary : public faiss::IndexBinary {
public:
    MOCK_METHOD(void, add, (idx_t n, const uint8_t* x), (override));
    MOCK_METHOD(void, search, (idx_t n, const uint8_t* x, idx_t k, int32_t* distances, idx_t* labels, const faiss::SearchParameters* params), (const, override));
    MOCK_METHOD(void, reset, (), (override));
};

#endif  // OPENSEARCH_KNN_FAISS_INDEX_MOCK_H