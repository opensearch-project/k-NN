// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

/**
 * This file contains classes for index operations which are free of JNI
 */

#ifndef OPENSEARCH_KNN_FAISS_INDEX_SERVICE_H
#define OPENSEARCH_KNN_FAISS_INDEX_SERVICE_H

#include <jni.h>
#include "faiss/MetricType.h"
#include "jni_util.h"
#include "faiss_methods.h"
#include <memory>

namespace knn_jni {
namespace faiss_wrapper {


/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class IndexService {
public:
    IndexService(std::unique_ptr<FaissMethods> faissMethods);
    //TODO Remove dependency on JNIUtilInterface and JNIEnv
    //TODO Reduce the number of parameters

    /**
     * Create index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numIds number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param vectorsAddress memory address which is holding vector data
     * @param ids a list of document ids for corresponding vectors
     * @param indexPath path to write index
     * @param parameters parameters to be applied to faiss index
     */
    virtual void createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters);
    virtual ~IndexService() = default;
protected:
    std::unique_ptr<FaissMethods> faissMethods;
};

/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class BinaryIndexService : public IndexService {
public:
    //TODO Remove dependency on JNIUtilInterface and JNIEnv
    //TODO Reduce the number of parameters
    BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods);
    /**
     * Create binary index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numIds number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param vectorsAddress memory address which is holding vector data
     * @param ids a list of document ids for corresponding vectors
     * @param indexPath path to write index
     * @param parameters parameters to be applied to faiss index
     */
    virtual void createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters
    ) override;
    virtual ~BinaryIndexService() = default;
};

}
}


#endif //OPENSEARCH_KNN_FAISS_INDEX_SERVICE_H
