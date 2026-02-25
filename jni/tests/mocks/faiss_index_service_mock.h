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

#ifndef OPENSEARCH_KNN_FAISS_INDEX_SERVICE_MOCK_H
#define OPENSEARCH_KNN_FAISS_INDEX_SERVICE_MOCK_H

#include "faiss_index_service.h"
#include <gmock/gmock.h>

using ::knn_jni::faiss_wrapper::FaissMethods;
using ::knn_jni::faiss_wrapper::IndexService;
typedef std::unordered_map<std::string, jobject> StringToJObjectMap;

class MockIndexService : public IndexService {
public:
    explicit MockIndexService(std::unique_ptr<FaissMethods> _faissMethods)
      : IndexService(std::move(_faissMethods)) {
    }

    MOCK_METHOD(
        long,
        initIndex,
        (
            knn_jni::JNIUtilInterface * jniUtil,
            JNIEnv * env,
            faiss::MetricType metric,
            std::string indexDescription,
            int dim,
            int numIds,
            int threadCount,
            StringToJObjectMap parameters
        ),
        (override));

    MOCK_METHOD(
        void,
        insertToIndex,
        (
            int dim,
            int numIds,
            int threadCount,
            int64_t vectorsAddress,
            std::vector<int64_t> & ids,
            long indexPtr
        ),
        (override));

    MOCK_METHOD(
        void,
        writeIndex,
        (
            faiss::IOWriter* writer,
            long indexPtr
        ),
        (override));
};

#endif  // OPENSEARCH_KNN_FAISS_INDEX_SERVICE_MOCK_H
