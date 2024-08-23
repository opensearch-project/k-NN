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
    /**
     * Initialize index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param parameters parameters to be applied to faiss index
     * @return memory address of the native index object
     */
    virtual jlong initIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, faiss::MetricType metric, std::string indexDescription, int dim, int numVectors, int threadCount, std::unordered_map<std::string, jobject> parameters);
    /**
     * Add vectors to index
     *
     * @param dim dimension of vectors
     * @param numIds number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param vectorsAddress memory address which is holding vector data
     * @param idMapAddress memory address of the native index object
     */
    virtual void insertToIndex(int dim, int numIds, int threadCount, int64_t vectorsAddress, std::vector<int64_t> &ids, jlong idMapAddress);
    /**
     * Write index to disk
     *
     * @param threadCount number of thread count to be used while adding data
     * @param indexPath path to write index
     * @param idMap memory address of the native index object
     */
    virtual void writeIndex(std::string indexPath, jlong idMapAddress);
    virtual ~IndexService() = default;
protected:
    virtual void allocIndex(faiss::Index * index, size_t dim, size_t numVectors);
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
     * Initialize index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param parameters parameters to be applied to faiss index
     * @return memory address of the native index object
     */
    virtual jlong initIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, faiss::MetricType metric, std::string indexDescription, int dim, int numVectors, int threadCount, std::unordered_map<std::string, jobject> parameters) override;
    /**
     * Add vectors to index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numIds number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param vectorsAddress memory address which is holding vector data
     * @param idMap a map of document id and vector id
     * @param parameters parameters to be applied to faiss index
     */
    virtual void insertToIndex(int dim, int numIds, int threadCount, int64_t vectorsAddress, std::vector<int64_t> &ids, jlong idMapAddress) override;
    /**
     * Write index to disk
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param threadCount number of thread count to be used while adding data
     * @param indexPath path to write index
     * @param idMap a map of document id and vector id
     * @param parameters parameters to be applied to faiss index
     */
    virtual void writeIndex(std::string indexPath, jlong idMapAddress) override;
    virtual ~BinaryIndexService() = default;
protected:
    virtual void allocIndex(faiss::Index * index, size_t dim, size_t numVectors) override;
};

/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class ByteIndexService : public IndexService {
public:
    //TODO Remove dependency on JNIUtilInterface and JNIEnv
    //TODO Reduce the number of parameters
    ByteIndexService(std::unique_ptr<FaissMethods> faissMethods);

/**
     * Initialize index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param parameters parameters to be applied to faiss index
     * @return memory address of the native index object
     */
    virtual jlong initIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, faiss::MetricType metric, std::string indexDescription, int dim, int numVectors, int threadCount, std::unordered_map<std::string, jobject> parameters) override;
    /**
     * Add vectors to index
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param dim dimension of vectors
     * @param numIds number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param vectorsAddress memory address which is holding vector data
     * @param idMap a map of document id and vector id
     * @param parameters parameters to be applied to faiss index
     */
    virtual void insertToIndex(int dim, int numIds, int threadCount, int64_t vectorsAddress, std::vector<int64_t> &ids, jlong idMapAddress) override;
    /**
     * Write index to disk
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param metric space type for distance calculation
     * @param indexDescription index description to be used by faiss index factory
     * @param threadCount number of thread count to be used while adding data
     * @param indexPath path to write index
     * @param idMap a map of document id and vector id
     * @param parameters parameters to be applied to faiss index
     */
    virtual void writeIndex(std::string indexPath, jlong idMapAddress) override;
    virtual ~ByteIndexService() = default;
protected:
    virtual void allocIndex(faiss::Index * index, size_t dim, size_t numVectors) override;
};

}
}


#endif //OPENSEARCH_KNN_FAISS_INDEX_SERVICE_H