// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_index_service.h"
#include "faiss_methods.h"
#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexIDMap.h"

#include <string>
#include <vector>
#include <memory>
#include <type_traits>

namespace knn_jni {
namespace faiss_wrapper {

template<typename INDEX, typename IVF, typename HNSW>
void SetExtraParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                        const std::unordered_map<std::string, jobject>& parametersCpp, INDEX * index) {
    std::unordered_map<std::string,jobject>::const_iterator value;
    if (auto * indexIvf = dynamic_cast<IVF*>(index)) {
        if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
            indexIvf->nprobe = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                && indexIvf->quantizer != nullptr) {
            auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, value->second);
            SetExtraParameters<INDEX, IVF, HNSW>(jniUtil, env, subParametersCpp, indexIvf->quantizer);
        }
    }

    if (auto * indexHnsw = dynamic_cast<HNSW*>(index)) {

        if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
            indexHnsw->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
            indexHnsw->hnsw.efSearch = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }
    }
}

IndexService::IndexService(std::unique_ptr<FaissMethods> _faissMethods) : faissMethods(std::move(_faissMethods)) {}

void IndexService::allocIndex(faiss::Index * index, size_t dim, size_t numVectors) {
    if (auto * indexHNSWSQ = dynamic_cast<faiss::IndexHNSWSQ *>(index)) {
        if (auto * indexScalarQuantizer = dynamic_cast<faiss::IndexScalarQuantizer *>(indexHNSWSQ->storage)) {
            indexScalarQuantizer->codes.reserve(indexScalarQuantizer->code_size * numVectors);
        }
    }
    if (auto * indexHNSW = dynamic_cast<faiss::IndexHNSW *>(index)) {
        if(auto * indexFlat = dynamic_cast<faiss::IndexFlat *>(indexHNSW->storage)) {
            indexFlat->codes.reserve(indexFlat->code_size * numVectors);
        }
    }
}

jlong IndexService::initIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numVectors,
        int threadCount,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Create index using Faiss factory method
    std::unique_ptr<faiss::Index> index(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, index.get());

    // Check that the index does not need to be trained
    if (!index->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    std::unique_ptr<faiss::IndexIDMap> idMap (faissMethods->indexIdMap(index.get()));
    //Makes sure the index is deleted when the destructor is called, this cannot be passed in the constructor
    idMap->own_fields = true;

    allocIndex(dynamic_cast<faiss::Index *>(idMap->index), dim, numVectors);

    //Release the ownership so as to make sure not delete the underlying index that is created. The index is needed later
    //in insert and write operations
    index.release();
    return reinterpret_cast<jlong>(idMap.release());
}

void IndexService::insertToIndex(
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> & ids,
        jlong idMapAddress
    ) {
    // Read vectors from memory address
    std::vector<float> * inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    faiss::IndexIDMap * idMap = reinterpret_cast<faiss::IndexIDMap *> (idMapAddress);

    // Add vectors
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void IndexService::updateIndexSettings(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        jlong idMapAddress,
        std::unordered_map<std::string, jobject> settings
    ) {

    faiss::IndexIDMap * idMap = reinterpret_cast<faiss::IndexIDMap *> (idMapAddress);

    for (const auto& entry : settings) {
        const std::string& key = entry.first;
        jobject value = entry.second;
        if (key == "base_level_only") {
            auto hnswCagra = dynamic_cast<faiss::IndexHNSWCagra*>(idMap->index);

            bool cppBool = jniUtil->ConvertJavaBoolToCppBool(env, value);

            if (hnswCagra) {
                hnswCagra->base_level_only = cppBool;
            }
        }
    }


}

void IndexService::writeIndex(
    faiss::IOWriter* writer,
    jlong idMapAddress
) {
    std::unique_ptr<faiss::IndexIDMap> idMap (reinterpret_cast<faiss::IndexIDMap *> (idMapAddress));

    try {
        // Write the index to disk
        faissMethods->writeIndex(idMap.get(), writer);
        if (auto openSearchIOWriter = dynamic_cast<knn_jni::stream::FaissOpenSearchIOWriter*>(writer)) {
            openSearchIOWriter->flush();
        }
    } catch(std::exception &e) {
        throw std::runtime_error("Failed to write index to disk");
    }
}

BinaryIndexService::BinaryIndexService(std::unique_ptr<FaissMethods> _faissMethods)
  : IndexService(std::move(_faissMethods)) {
}

void BinaryIndexService::allocIndex(faiss::Index * index, size_t dim, size_t numVectors) {
    if (auto * indexBinaryHNSW = dynamic_cast<faiss::IndexBinaryHNSW *>(index)) {
        auto * indexBinaryFlat = dynamic_cast<faiss::IndexBinaryFlat *>(indexBinaryHNSW->storage);
        indexBinaryFlat->xb.reserve(dim * numVectors / 8);
    }
}

jlong BinaryIndexService::initIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numVectors,
        int threadCount,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Create index using Faiss factory method
    std::unique_ptr<faiss::IndexBinary> index(faissMethods->indexBinaryFactory(dim, indexDescription.c_str()));
    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
       omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, parameters, index.get());

    // Check that the index does not need to be trained
    if (!index->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(index.get()));
    //Makes sure the index is deleted when the destructor is called
    idMap->own_fields = true;

    allocIndex(dynamic_cast<faiss::Index *>(idMap->index), dim, numVectors);

    //Release the ownership so as to make sure not delete the underlying index that is created. The index is needed later
    //in insert and write operations
    index.release();
    return reinterpret_cast<jlong>(idMap.release());
}

void BinaryIndexService::insertToIndex(
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> & ids,
        jlong idMapAddress
    ) {
    // Read vectors from memory address (unique ptr since we want to remove from memory after use)
    std::vector<uint8_t> * inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    faiss::IndexBinaryIDMap * idMap = reinterpret_cast<faiss::IndexBinaryIDMap *> (idMapAddress);

    // Add vectors
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void BinaryIndexService::writeIndex(
    faiss::IOWriter* writer,
    jlong idMapAddress
) {
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap (reinterpret_cast<faiss::IndexBinaryIDMap *> (idMapAddress));

    try {
        // Write the index to disk
        faissMethods->writeIndexBinary(idMap.get(), writer);
        if (auto openSearchIOWriter = dynamic_cast<knn_jni::stream::FaissOpenSearchIOWriter*>(writer)) {
            openSearchIOWriter->flush();
        }
    } catch(std::exception &e) {
        throw std::runtime_error("Failed to write index to disk");
    }
}

ByteIndexService::ByteIndexService(std::unique_ptr<FaissMethods> _faissMethods)
  : IndexService(std::move(_faissMethods)) {
}

void ByteIndexService::allocIndex(faiss::Index * index, size_t dim, size_t numVectors) {
    if (auto * indexHNSWSQ = dynamic_cast<faiss::IndexHNSWSQ *>(index)) {
        if(auto * indexScalarQuantizer = dynamic_cast<faiss::IndexScalarQuantizer *>(indexHNSWSQ->storage)) {
            indexScalarQuantizer->codes.reserve(indexScalarQuantizer->code_size * numVectors);
        }
    }
}

jlong ByteIndexService::initIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numVectors,
        int threadCount,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Create index using Faiss factory method
    std::unique_ptr<faiss::Index> index(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, index.get());

    // Check that the index does not need to be trained
    if(!index->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    std::unique_ptr<faiss::IndexIDMap> idMap (faissMethods->indexIdMap(index.get()));
    //Makes sure the index is deleted when the destructor is called, this cannot be passed in the constructor
    idMap->own_fields = true;

    allocIndex(dynamic_cast<faiss::Index *>(idMap->index), dim, numVectors);

    //Release the ownership so as to make sure not delete the underlying index that is created. The index is needed later
    //in insert and write operations
    index.release();
    return reinterpret_cast<jlong>(idMap.release());
}

void ByteIndexService::insertToIndex(
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> & ids,
        jlong idMapAddress
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<int8_t>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = inputVectors->size() / dim;
    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    faiss::IndexIDMap * idMap = reinterpret_cast<faiss::IndexIDMap *> (idMapAddress);

    // Add vectors in batches by casting int8 vectors into float with a batch size of 1000 to avoid additional memory spike.
    // Refer to this github issue for more details https://github.com/opensearch-project/k-NN/issues/1659#issuecomment-2307390255
    int batchSize = 1000;
    std::vector <float> inputFloatVectors(batchSize * dim);
    std::vector <int64_t> floatVectorsIds(batchSize);
    auto iter = inputVectors->begin();

    for (int id = 0; id < numVectors; id += batchSize) {
        if (numVectors - id < batchSize) {
            batchSize = numVectors - id;
        }

        for (int i = 0; i < batchSize; ++i) {
            floatVectorsIds[i] = ids[id + i];
            for (int j = 0; j < dim; ++j, ++iter) {
                inputFloatVectors[i * dim + j] = static_cast<float>(*iter);
            }
        }
        idMap->add_with_ids(batchSize, inputFloatVectors.data(), floatVectorsIds.data());
    }
}

void ByteIndexService::writeIndex(
    faiss::IOWriter* writer,
    jlong idMapAddress
) {
    std::unique_ptr<faiss::IndexIDMap> idMap (reinterpret_cast<faiss::IndexIDMap *> (idMapAddress));

    try {
        // Write the index to disk
        faissMethods->writeIndex(idMap.get(), writer);
        if (auto openSearchIOWriter = dynamic_cast<knn_jni::stream::FaissOpenSearchIOWriter*>(writer)) {
            openSearchIOWriter->flush();
        }
    } catch(std::exception &e) {
        throw std::runtime_error("Failed to write index to disk");
    }
}
} // namespace faiss_wrapper
} // namesapce knn_jni
