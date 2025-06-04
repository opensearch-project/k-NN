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
#include "faiss/impl/io.h"
#include "jni_util.h"
#include "faiss_methods.h"
#include "faiss_stream_support.h"
#include <memory>

namespace knn_jni {
namespace faiss_wrapper {


/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class IndexService {
public:
    explicit IndexService(std::unique_ptr<FaissMethods> faissMethods);

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
     * @param writer IOWriter implementation doing IO processing.
     *               In most cases, it is expected to have underlying Lucene's IndexOuptut.
     * @param idMapAddress memory address of the native index object
     */
    virtual void writeIndex(faiss::IOWriter* writer, jlong idMapAddress);

    /**
     * Initialize index from template
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param templateIndexJ template index
     * @return memory address of the native index object
     */
    virtual jlong initIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, int dim, int numVectors, int threadCount, jbyteArray templateIndexJ);


    virtual ~IndexService() = default;

protected:
    virtual void allocIndex(faiss::Index * index, size_t dim, size_t numVectors);
    virtual jlong initAndAllocateIndex(std::unique_ptr<faiss::Index> &index, size_t threadCount, size_t dim, size_t numVectors);

    std::unique_ptr<FaissMethods> faissMethods;
};  // class IndexService

/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class BinaryIndexService final : public IndexService {
public:
    //TODO Remove dependency on JNIUtilInterface and JNIEnv
    //TODO Reduce the number of parameters
    explicit BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods);

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
    jlong initIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, faiss::MetricType metric, std::string indexDescription, int dim, int numVectors, int threadCount, std::unordered_map<std::string, jobject> parameters) final;

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
    void insertToIndex(int dim, int numIds, int threadCount, int64_t vectorsAddress, std::vector<int64_t> &ids, jlong idMapAddress) final;

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
    void writeIndex(faiss::IOWriter* writer, jlong idMapAddress) final;

    /**
     * Initialize index from template
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param templateIndexJ template index
     * @return memory address of the native index object
     */
    jlong initIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, int dim, int numVectors, int threadCount, jbyteArray templateIndexJ) final;

protected:
    void allocIndex(faiss::Index * index, size_t dim, size_t numVectors) final;
    jlong initAndAllocateIndex(std::unique_ptr<faiss::IndexBinary> &index, size_t threadCount, size_t dim, size_t numVectors);
};  // class BinaryIndexService

/**
 * A class to provide operations on index
 * This class should evolve to have only cpp object but not jni object
 */
class ByteIndexService final : public IndexService {
public:
    //TODO Remove dependency on JNIUtilInterface and JNIEnv
    //TODO Reduce the number of parameters
    explicit ByteIndexService(std::unique_ptr<FaissMethods> faissMethods);

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
    jlong initIndex(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, faiss::MetricType metric, std::string indexDescription, int dim, int numVectors, int threadCount, std::unordered_map<std::string, jobject> parameters) final;

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
    void insertToIndex(int dim, int numIds, int threadCount, int64_t vectorsAddress, std::vector<int64_t> &ids, jlong idMapAddress) final;

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
    void writeIndex(faiss::IOWriter* writer, jlong idMapAddress) final;

    /**
     * Initialize index from template
     *
     * @param jniUtil jni util
     * @param env jni environment
     * @param dim dimension of vectors
     * @param numVectors number of vectors
     * @param threadCount number of thread count to be used while adding data
     * @param templateIndexJ template index
     * @return memory address of the native index object
     */
    jlong initIndexFromTemplate(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, int dim, int numVectors, int threadCount, jbyteArray templateIndexJ) final;

 protected:
    void allocIndex(faiss::Index * index, size_t dim, size_t numVectors) final;
    jlong initAndAllocateIndex(std::unique_ptr<faiss::Index> &index, size_t threadCount, size_t dim, size_t numVectors) final;
};  // class ByteIndexService

}
}


#endif //OPENSEARCH_KNN_FAISS_INDEX_SERVICE_H
