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
#include "faiss/index_factory.h"
#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <faiss/impl/io.h>

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

IndexService::IndexService(std::unique_ptr<FaissMethods> faissMethods) : faissMethods(std::move(faissMethods)) {}

void IndexService::createIndex(
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
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::Index> indexWriter(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexIDMap> idMap(faissMethods->indexIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndex(idMap.get(), indexPath.c_str());
}

void IndexService::createIndexFromTemplate(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        int dim,
        int numIds,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters,
        std::vector<uint8_t> templateIndexData
    ) {
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = templateIndexData;

    std::unique_ptr<faiss::Index> indexWriter(faissMethods->readIndex(&vectorIoReader, 0));

    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of vectors or IDs does not match expected values");
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, indexWriter.get());

    std::unique_ptr<faiss::IndexIDMap> idMap(faissMethods->indexIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    faissMethods->writeIndex(idMap.get(), indexPath.c_str());
}

void IndexService::InternalTrainIndex(faiss::Index * index, faiss::idx_t n, const float* x) {
    if (auto * indexIvf = dynamic_cast<faiss::IndexIVF*>(index)) {
        if (indexIvf->quantizer_trains_alone == 2) {
            InternalTrainIndex(indexIvf->quantizer, n, x);
        }
        indexIvf->make_direct_map();
    }

    if (!index->is_trained) {
        index->train(n, x);
    }
}

std::vector<uint8_t> IndexService::trainIndex(JNIUtilInterface* jniUtil, JNIEnv* env, faiss::MetricType metric, std::string& indexDescription, int dimension, int numVectors, float* trainingVectors, std::unordered_map<std::string, jobject>& parameters) {
    // Create faiss index
    std::unique_ptr<faiss::Index> index(faissMethods->indexFactory(dimension, indexDescription.c_str(), metric));

    // Train index if needed
    if (!index->is_trained) {
        InternalTrainIndex(index.get(), numVectors, trainingVectors);
    }

    // Write index to a vector
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index(index.get(), &vectorIoWriter);

    return std::vector<uint8_t>(vectorIoWriter.data.begin(), vectorIoWriter.data.end());
}



BinaryIndexService::BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods) : IndexService(std::move(faissMethods)) {}

void BinaryIndexService::createIndex(
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
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);

    if (dim % 8 != 0) {
        throw std::runtime_error("Dimensions should be multiply of 8");
    }
    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::IndexBinary> indexWriter(faissMethods->indexBinaryFactory(dim, indexDescription.c_str()));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndexBinary(idMap.get(), indexPath.c_str());
}

void BinaryIndexService::createIndexFromTemplate(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        int dim,
        int numIds,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters,
        std::vector<uint8_t> templateIndexData
    ) {
    faiss::VectorIOReader vectorIoReader;
    vectorIoReader.data = templateIndexData;

    std::unique_ptr<faiss::IndexBinary> indexWriter(faissMethods->readIndexBinary(&vectorIoReader, 0));

    auto *inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    if (numIds != numVectors) {
        throw std::runtime_error("Number of vectors or IDs does not match expected values");
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, parameters, indexWriter.get());

    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    faissMethods->writeIndexBinary(idMap.get(), indexPath.c_str());
}

void BinaryIndexService::InternalTrainIndex(faiss::IndexBinary * index, faiss::idx_t n, const float* x) {
    if (auto * indexIvf = dynamic_cast<faiss::IndexBinaryIVF*>(index)) {
        if (!indexIvf->is_trained) {
            indexIvf->train(n, reinterpret_cast<const uint8_t*>(x));
        }
    }
    if (!index->is_trained) {
        index->train(n, reinterpret_cast<const uint8_t*>(x));
    }
}

std::vector<uint8_t> BinaryIndexService::trainIndex(JNIUtilInterface* jniUtil, JNIEnv* env, faiss::MetricType metric, std::string& indexDescription, int dimension, int numVectors, float* trainingVectors, std::unordered_map<std::string, jobject>& parameters) {
    // Convert Java parameters to C++ parameters
    std::unique_ptr<faiss::IndexBinary> indexWriter;
    indexWriter.reset(faiss::index_binary_factory(dimension, indexDescription.c_str()));

    // Train the index if it is not already trained
    if (!indexWriter->is_trained) {
        InternalTrainIndex(indexWriter.get(), numVectors, trainingVectors);
    }

    // Serialize the trained index to a byte array
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index_binary(indexWriter.get(), &vectorIoWriter);

    // Convert the serialized data to a std::vector<uint8_t>
    std::vector<uint8_t> trainedIndexData(vectorIoWriter.data.begin(), vectorIoWriter.data.end());

    return trainedIndexData;
}
} // namespace faiss_wrapper
} // namespace knn_jni
