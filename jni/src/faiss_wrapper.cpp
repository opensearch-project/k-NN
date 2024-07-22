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

#include "jni_util.h"
#include "faiss_wrapper.h"
#include "faiss_util.h"
#include "faiss_index_service.h"

#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetaIndexes.h"
#include "faiss/Index.h"
#include "faiss/impl/IDSelector.h"
#include "faiss/IndexIVFPQ.h"
#include "commons.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexBinaryHNSW.h"

#include <algorithm>
#include <jni.h>
#include <string>
#include <vector>

// Defines type of IDSelector
enum FilterIdsSelectorType{
    BITMAP = 0, BATCH = 1,
};
namespace faiss {

// Using jlong to do Bitmap selector, jlong[] equals to lucene FixedBitSet#bits
struct IDSelectorJlongBitmap : IDSelector {
    size_t n;
    const jlong* bitmap;

    /** Construct with a binary mask like Lucene FixedBitSet
     *
     * @param n size of the bitmap array
     * @param bitmap id like Lucene FixedBitSet bits
     */
    IDSelectorJlongBitmap(size_t n, const jlong* bitmap) : n(n), bitmap(bitmap) {};
    bool is_member(idx_t id) const final {
        uint64_t index = id;
        uint64_t i = index >> 6;  // div 64
        if (i >= n ) {
            return false;
        }
        return (bitmap[i] >> ( index & 63)) & 1L;
    }
    ~IDSelectorJlongBitmap() override {}
};
}
// Translate space type to faiss metric
faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType);

// Set additional parameters on faiss index
void SetExtraParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                        const std::unordered_map<std::string, jobject>& parametersCpp, faiss::Index * index);

// Train an index with data provided
void InternalTrainIndex(faiss::Index * index, faiss::idx_t n, const float* x);

// Train a binary index with data provided
void InternalTrainBinaryIndex(faiss::IndexBinary * index, faiss::idx_t n, const float* x);

// Converts the int FilterIds to Faiss ids type array.
void convertFilterIdsToFaissIdType(const int* filterIds, int filterIdsLength, faiss::idx_t* convertedFilterIds);

// Concerts the FilterIds to BitMap
void buildFilterIdsBitMap(const int* filterIds, int filterIdsLength, uint8_t* bitsetVector);

std::unique_ptr<faiss::IDGrouperBitmap> buildIDGrouperBitmap(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env, jintArray parentIdsJ, std::vector<uint64_t>* bitmap);

// Check if a loaded index is an IVFPQ index with l2 space type
bool isIndexIVFPQL2(faiss::Index * index);

// Gets IVFPQ index from a faiss index. For faiss, we wrap the index in the type
// IndexIDMap which has member that will point to underlying index that stores the data
faiss::IndexIVFPQ * extractIVFPQIndex(faiss::Index * index);

void knn_jni::faiss_wrapper::CreateIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ, jlong vectorsAddressJ, jint dimJ,
                                         jstring indexPathJ, jobject parametersJ, IndexService* indexService) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }

    if(dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    // parametersJ is a Java Map<String, Object>. ConvertJavaMapToCppMap converts it to a c++ map<string, jobject>
    // so that it is easier to access.
    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    // Parameters to pass
    // Metric type
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);
    jniUtil->DeleteLocalRef(env, spaceTypeJ);

    // Dimension
    int dim = (int)dimJ;

    // Number of vectors
    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);

    // Index description
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));
    jniUtil->DeleteLocalRef(env, indexDescriptionJ);

    // Thread count
    int threadCount = 0;
    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
    }

    // Vectors address
    int64_t vectorsAddress = (int64_t)vectorsAddressJ;

    // Ids
    auto ids = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);

    // Index path
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));

    // Extra parameters
    // TODO: parse the entire map and remove jni object
    std::unordered_map<std::string, jobject> subParametersCpp;
    if(parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersCpp[knn_jni::PARAMETERS]);
    }
    // end parameters to pass

    // Create index
    indexService->createIndex(jniUtil, env, metric, indexDescriptionCpp, dim, numIds, threadCount, vectorsAddress, ids, indexPathCpp, subParametersCpp);
}

void knn_jni::faiss_wrapper::CreateIndexFromTemplate(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                                     jlong vectorsAddressJ, jint dimJ, jstring indexPathJ,
                                                     jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }

    if(dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    if (templateIndexJ == nullptr) {
        throw std::runtime_error("Template index cannot be null");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);
    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        auto threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
        omp_set_num_threads(threadCount);
    }
    jniUtil->DeleteLocalRef(env, parametersJ);

    // Read data set
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
    int dim = (int)dimJ;
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Get vector of bytes from jbytearray
    int indexBytesCount = jniUtil->GetJavaBytesArrayLength(env, templateIndexJ);
    jbyte * indexBytesJ = jniUtil->GetByteArrayElements(env, templateIndexJ, nullptr);

    faiss::VectorIOReader vectorIoReader;
    for (int i = 0; i < indexBytesCount; i++) {
        vectorIoReader.data.push_back((uint8_t) indexBytesJ[i]);
    }
    jniUtil->ReleaseByteArrayElements(env, templateIndexJ, indexBytesJ, JNI_ABORT);

    // Create faiss index
    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::read_index(&vectorIoReader, 0));

    auto idVector = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);
    faiss::IndexIDMap idMap =  faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, inputVectors->data(), idVector.data());
    // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
    // This is not the ideal approach, please refer this gh issue for long term solution:
    // https://github.com/opensearch-project/k-NN/issues/1600
    delete inputVectors;
    // Write the index to disk
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index(&idMap, indexPathCpp.c_str());
}

void knn_jni::faiss_wrapper::CreateBinaryIndexFromTemplate(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                                     jlong vectorsAddressJ, jint dimJ, jstring indexPathJ,
                                                     jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }

    if(dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    if (templateIndexJ == nullptr) {
        throw std::runtime_error("Template index cannot be null");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);
    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        auto threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
        omp_set_num_threads(threadCount);
    }
    jniUtil->DeleteLocalRef(env, parametersJ);

    // Read data set
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddressJ);
    int dim = (int)dimJ;
    if (dim % 8 != 0) {
        throw std::runtime_error("Dimensions should be multiply of 8");
    }
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Get vector of bytes from jbytearray
    int indexBytesCount = jniUtil->GetJavaBytesArrayLength(env, templateIndexJ);
    jbyte * indexBytesJ = jniUtil->GetByteArrayElements(env, templateIndexJ, nullptr);

    faiss::VectorIOReader vectorIoReader;
    for (int i = 0; i < indexBytesCount; i++) {
        vectorIoReader.data.push_back((uint8_t) indexBytesJ[i]);
    }
    jniUtil->ReleaseByteArrayElements(env, templateIndexJ, indexBytesJ, JNI_ABORT);

    // Create faiss index
    std::unique_ptr<faiss::IndexBinary> indexWriter;
    indexWriter.reset(faiss::read_index_binary(&vectorIoReader, 0));

    auto idVector = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);
    faiss::IndexBinaryIDMap idMap =  faiss::IndexBinaryIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, reinterpret_cast<const uint8_t*>(inputVectors->data()), idVector.data());
    // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
    // This is not the ideal approach, please refer this gh issue for long term solution:
    // https://github.com/opensearch-project/k-NN/issues/1600
    delete inputVectors;
    // Write the index to disk
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index_binary(&idMap, indexPathCpp.c_str());
}

jlong knn_jni::faiss_wrapper::LoadIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    // Skipping IO_FLAG_PQ_SKIP_SDC_TABLE because the index is read only and the sdc table is only used during ingestion
    // Skipping IO_PRECOMPUTE_TABLE because it is only needed for IVFPQ-l2 and it leads to high memory consumption if
    // done for each segment. Instead, we will set it later on with `setSharedIndexState`
    faiss::Index* indexReader = faiss::read_index(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY | faiss::IO_FLAG_PQ_SKIP_SDC_TABLE | faiss::IO_FLAG_SKIP_PRECOMPUTE_TABLE);
    return (jlong) indexReader;
}

jlong knn_jni::faiss_wrapper::LoadBinaryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    // Skipping IO_FLAG_PQ_SKIP_SDC_TABLE because the index is read only and the sdc table is only used during ingestion
    // Skipping IO_PRECOMPUTE_TABLE because it is only needed for IVFPQ-l2 and it leads to high memory consumption if
    // done for each segment. Instead, we will set it later on with `setSharedIndexState`
    faiss::IndexBinary* indexReader = faiss::read_index_binary(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY | faiss::IO_FLAG_PQ_SKIP_SDC_TABLE | faiss::IO_FLAG_SKIP_PRECOMPUTE_TABLE);
    return (jlong) indexReader;
}

bool knn_jni::faiss_wrapper::IsSharedIndexStateRequired(jlong indexPointerJ) {
    auto * index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    return isIndexIVFPQL2(index);
}

jlong knn_jni::faiss_wrapper::InitSharedIndexState(jlong indexPointerJ) {
    auto * index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    if (!isIndexIVFPQL2(index)) {
        throw std::runtime_error("Unable to init shared index state from index. index is not of type IVFPQ-l2");
    }

    auto * indexIVFPQ = extractIVFPQIndex(index);
    int use_precomputed_table = 0;
    auto * sharedMemoryAddress = new faiss::AlignedTable<float>();
    faiss::initialize_IVFPQ_precomputed_table(
            use_precomputed_table,
            indexIVFPQ->quantizer,
            indexIVFPQ->pq,
            *sharedMemoryAddress,
            indexIVFPQ->by_residual,
            indexIVFPQ->verbose);
    return (jlong) sharedMemoryAddress;
}

void knn_jni::faiss_wrapper::SetSharedIndexState(jlong indexPointerJ, jlong shareIndexStatePointerJ) {
    auto * index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    if (!isIndexIVFPQL2(index)) {
        throw std::runtime_error("Unable to set shared index state from index. index is not of type IVFPQ-l2");
    }
    auto * indexIVFPQ = extractIVFPQIndex(index);

    //TODO: Currently, the only shared state is that of the AlignedTable associated with
    // IVFPQ-l2 index type (see https://github.com/opensearch-project/k-NN/issues/1507). In the future,
    // this will be generalized and more information will be needed to determine the shared type. But, until then,
    // this is fine.
    auto *alignTable = reinterpret_cast<faiss::AlignedTable<float>*>(shareIndexStatePointerJ);
    // In faiss, usePrecomputedTable can have a couple different values:
    //  -1  -> dont use the table
    //   0  -> tell initialize_IVFPQ_precomputed_table to select the best value and change the value
    //   1  -> default behavior
    //   2  -> Index is of type "MultiIndexQuantizer"
    // This index will be of type IndexIVFPQ always. We never create "MultiIndexQuantizer". So, the value we
    // want is 1.
    // (ref: https://github.com/facebookresearch/faiss/blob/v1.8.0/faiss/IndexIVFPQ.cpp#L383-L410)
    int usePrecomputedTable = 1;
    indexIVFPQ->set_precomputed_table(alignTable, usePrecomputedTable);
}

jobjectArray knn_jni::faiss_wrapper::QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jintArray parentIdsJ) {
    return knn_jni::faiss_wrapper::QueryIndex_WithFilter(jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, nullptr, 0, parentIdsJ);
}

jobjectArray knn_jni::faiss_wrapper::QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }
    // The ids vector will hold the top k ids from the search and the dis vector will hold the top k distances from
    // the query point
    std::vector<float> dis(kJ);
    std::vector<faiss::idx_t> ids(kJ);
    float* rawQueryvector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
    /*
        Setting the omp_set_num_threads to 1 to make sure that no new OMP threads are getting created.
    */
    omp_set_num_threads(1);
    // create the filterSearch params if the filterIdsJ is not a null pointer
    if(filterIdsJ != nullptr) {
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        std::unique_ptr<faiss::IDSelector> idSelector;
        if(filterIdsTypeJ == BITMAP) {
            idSelector.reset(new faiss::IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
        } else {
            faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
            idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
        }
        faiss::SearchParameters *searchParameters;
        faiss::SearchParametersHNSW hnswParams;
        faiss::SearchParametersIVF ivfParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader) {
            // Query param efsearch supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            hnswParams.sel = idSelector.get();
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        } else {
            auto ivfReader = dynamic_cast<const faiss::IndexIVF*>(indexReader->index);
            auto ivfFlatReader = dynamic_cast<const faiss::IndexIVFFlat*>(indexReader->index);
            
            if(ivfReader || ivfFlatReader) {
                int indexNprobe = ivfReader == nullptr ? ivfFlatReader->nprobe : ivfReader->nprobe;
                ivfParams.nprobe = commons::getIntegerMethodParameter(env, jniUtil, methodParams, NPROBES, indexNprobe);
                ivfParams.sel = idSelector.get();
                searchParameters = &ivfParams;
            }
        }
        try {
            indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
    } else {
        faiss::SearchParameters *searchParameters = nullptr;
        faiss::SearchParametersHNSW hnswParams;
        faiss::SearchParametersIVF ivfParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader != nullptr) {
            // Query param efsearch supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        } else {
            auto ivfReader = dynamic_cast<const faiss::IndexIVF*>(indexReader->index);
            if (ivfReader) {
                int indexNprobe = ivfReader->nprobe;
                ivfParams.nprobe = commons::getIntegerMethodParameter(env, jniUtil, methodParams, NPROBES, indexNprobe);
                searchParameters = &ivfParams;
            }
        }
        try {
            indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            throw;
        }
    }
    jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);

    // If there are not k results, the results will be padded with -1. Find the first -1, and set result size to that
    // index
    int resultSize = kJ;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }

    jclass resultClass = jniUtil->FindClass(env,"org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);

    jobject result;
    for(int i = 0; i < resultSize; ++i) {
        result = jniUtil->NewObject(env, resultClass, allArgs, ids[i], dis[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
    }
    return results;
}

jobjectArray knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                jbyteArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexBinaryIDMap *>(indexPointerJ);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    // The ids vector will hold the top k ids from the search and the dis vector will hold the top k distances from
    // the query point
    std::vector<int32_t> dis(kJ);
    std::vector<faiss::idx_t> ids(kJ);
    int8_t* rawQueryvector = jniUtil->GetByteArrayElements(env, queryVectorJ, nullptr);
    /*
        Setting the omp_set_num_threads to 1 to make sure that no new OMP threads are getting created.
    */
    omp_set_num_threads(1);
    // create the filterSearch params if the filterIdsJ is not a null pointer
    if(filterIdsJ != nullptr) {
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        std::unique_ptr<faiss::IDSelector> idSelector;
        if(filterIdsTypeJ == BITMAP) {
            idSelector.reset(new faiss::IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
        } else {
            faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
            idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
        }
        faiss::SearchParameters *searchParameters;
        faiss::SearchParametersHNSW hnswParams;
        faiss::SearchParametersIVF ivfParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexBinaryHNSW*>(indexReader->index);
        if(hnswReader) {
            // Query param efsearch supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            hnswParams.sel = idSelector.get();
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        } else {
            auto ivfReader = dynamic_cast<const faiss::IndexBinaryIVF*>(indexReader->index);
            if(ivfReader) {
                ivfParams.sel = idSelector.get();
                searchParameters = &ivfParams;
            }
        }
        try {
            indexReader->search(1, reinterpret_cast<uint8_t*>(rawQueryvector), kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseByteArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
    } else {
        faiss::SearchParameters *searchParameters = nullptr;
        faiss::SearchParametersHNSW hnswParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexBinaryHNSW*>(indexReader->index);
        // TODO currently, search parameter is not supported in binary index
        // To avoid test failure, we skip setting ef search when methodPramsJ is null temporary
        if(hnswReader!= nullptr && (methodParamsJ != nullptr || parentIdsJ != nullptr)) {
            // Query param efsearch supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        }
        try {
            indexReader->search(1, reinterpret_cast<uint8_t*>(rawQueryvector), kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseByteArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            throw;
        }
    }
    jniUtil->ReleaseByteArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);

    // If there are not k results, the results will be padded with -1. Find the first -1, and set result size to that
    // index
    int resultSize = kJ;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }

    jclass resultClass = jniUtil->FindClass(env,"org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);

    jobject result;
    for(int i = 0; i < resultSize; ++i) {
        result = jniUtil->NewObject(env, resultClass, allArgs, ids[i], dis[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
    }
    return results;
}

void knn_jni::faiss_wrapper::Free(jlong indexPointer, jboolean isBinaryIndexJ) {
    bool isBinaryIndex = static_cast<bool>(isBinaryIndexJ);
    if (isBinaryIndex) {
        auto *indexWrapper = reinterpret_cast<faiss::IndexBinary*>(indexPointer);
        delete indexWrapper;
    }
    else {
        auto *indexWrapper = reinterpret_cast<faiss::Index*>(indexPointer);
        delete indexWrapper;
    }
}

void knn_jni::faiss_wrapper::FreeSharedIndexState(jlong shareIndexStatePointerJ) {
    //TODO: Currently, the only shared state is that of the AlignedTable associated with
    // IVFPQ-l2 index type (see https://github.com/opensearch-project/k-NN/issues/1507). In the future,
    // this will be generalized and more information will be needed to determine the shared type. But, until then,
    // this is fine.
    auto *alignTable = reinterpret_cast<faiss::AlignedTable<float>*>(shareIndexStatePointerJ);
    delete alignTable;
}

void knn_jni::faiss_wrapper::InitLibrary() {
    //set thread 1 cause ES has Search thread
    //TODO make it different at search and write
    //	omp_set_num_threads(1);
}

jbyteArray knn_jni::faiss_wrapper::TrainIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject parametersJ,
                                              jint dimensionJ, jlong trainVectorsPointerJ) {
    // First, we need to build the index
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    // Create faiss index
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));

    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::index_factory((int) dimensionJ, indexDescriptionCpp.c_str(), metric));

    // Related to https://github.com/facebookresearch/faiss/issues/1621. HNSWPQ defaults to l2 even when metric is
    // passed in. This updates it to the correct metric.
    indexWriter->metric_type = metric;
    if (auto * indexHnswPq = dynamic_cast<faiss::IndexHNSWPQ*>(indexWriter.get())) {
        indexHnswPq->storage->metric_type = metric;
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        auto threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    if(parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        jobject subParametersJ = parametersCpp[knn_jni::PARAMETERS];
        auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, subParametersJ);
        SetExtraParameters(jniUtil, env, subParametersCpp, indexWriter.get());
        jniUtil->DeleteLocalRef(env, subParametersJ);
    }

    // Train index if needed
    auto *trainingVectorsPointerCpp = reinterpret_cast<std::vector<float>*>(trainVectorsPointerJ);
    int numVectors = trainingVectorsPointerCpp->size()/(int) dimensionJ;
    if(!indexWriter->is_trained) {
        InternalTrainIndex(indexWriter.get(), numVectors, trainingVectorsPointerCpp->data());
    }
    jniUtil->DeleteLocalRef(env, parametersJ);

    // Now that indexWriter is trained, we just load the bytes into an array and return
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index(indexWriter.get(), &vectorIoWriter);

    // Wrap in smart pointer
    std::unique_ptr<jbyte[]> jbytesBuffer;
    jbytesBuffer.reset(new jbyte[vectorIoWriter.data.size()]);
    int c = 0;
    for (auto b : vectorIoWriter.data) {
        jbytesBuffer[c++] = (jbyte) b;
    }

    jbyteArray ret = jniUtil->NewByteArray(env, vectorIoWriter.data.size());
    jniUtil->SetByteArrayRegion(env, ret, 0, vectorIoWriter.data.size(), jbytesBuffer.get());
    return ret;
}

jbyteArray knn_jni::faiss_wrapper::TrainBinaryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject parametersJ,
                                              jint dimensionJ, jlong trainVectorsPointerJ) {
    // First, we need to build the index
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    // Create faiss index
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));

    std::unique_ptr<faiss::IndexBinary> indexWriter;
    indexWriter.reset(faiss::index_binary_factory((int) dimensionJ, indexDescriptionCpp.c_str()));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        auto threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
        omp_set_num_threads(threadCount);
    }

    // Train index if needed
    auto *trainingVectorsPointerCpp = reinterpret_cast<std::vector<float>*>(trainVectorsPointerJ);
    int numVectors = trainingVectorsPointerCpp->size()/(int) dimensionJ;
    if(!indexWriter->is_trained) {
        InternalTrainBinaryIndex(indexWriter.get(), numVectors, trainingVectorsPointerCpp->data());
    }
    jniUtil->DeleteLocalRef(env, parametersJ);

    // Now that indexWriter is trained, we just load the bytes into an array and return
    faiss::VectorIOWriter vectorIoWriter;
    faiss::write_index_binary(indexWriter.get(), &vectorIoWriter);

    // Wrap in smart pointer
    std::unique_ptr<jbyte[]> jbytesBuffer;
    jbytesBuffer.reset(new jbyte[vectorIoWriter.data.size()]);
    int c = 0;
    for (auto b : vectorIoWriter.data) {
        jbytesBuffer[c++] = (jbyte) b;
    }

    jbyteArray ret = jniUtil->NewByteArray(env, vectorIoWriter.data.size());
    jniUtil->SetByteArrayRegion(env, ret, 0, vectorIoWriter.data.size(), jbytesBuffer.get());
    return ret;
}

faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType) {
    if (spaceType == knn_jni::L2) {
        return faiss::METRIC_L2;
    }

    if (spaceType == knn_jni::INNER_PRODUCT) {
        return faiss::METRIC_INNER_PRODUCT;
    }

    // Space type is not used for binary index. Use L2 just to avoid an error.
    if (spaceType == knn_jni::HAMMING) {
        return faiss::METRIC_L2;
    }

    throw std::runtime_error("Invalid spaceType");
}

void SetExtraParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                        const std::unordered_map<std::string, jobject>& parametersCpp, faiss::Index * index) {

    std::unordered_map<std::string,jobject>::const_iterator value;
    if (auto * indexIvf = dynamic_cast<faiss::IndexIVF*>(index)) {
        if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
            indexIvf->nprobe = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                && indexIvf->quantizer != nullptr) {
            auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, value->second);
            SetExtraParameters(jniUtil, env, subParametersCpp, indexIvf->quantizer);
        }
    }

    if (auto * indexHnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {

        if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
            indexHnsw->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
            indexHnsw->hnsw.efSearch = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }
    }
}

void InternalTrainIndex(faiss::Index * index, faiss::idx_t n, const float* x) {
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

void InternalTrainBinaryIndex(faiss::IndexBinary * index, faiss::idx_t n, const float* x) {
    if (auto * indexIvf = dynamic_cast<faiss::IndexBinaryIVF*>(index)) {
        indexIvf->make_direct_map();
    }
    if (!index->is_trained) {
        index->train(n, reinterpret_cast<const uint8_t*>(x));
    }
}

std::unique_ptr<faiss::IDGrouperBitmap> buildIDGrouperBitmap(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env, jintArray parentIdsJ, std::vector<uint64_t>* bitmap) {
    int *parentIdsArray = jniUtil->GetIntArrayElements(env, parentIdsJ, nullptr);
    int parentIdsLength = jniUtil->GetJavaIntArrayLength(env, parentIdsJ);
    std::unique_ptr<faiss::IDGrouperBitmap> idGrouper = faiss_util::buildIDGrouperBitmap(parentIdsArray, parentIdsLength, bitmap);
    jniUtil->ReleaseIntArrayElements(env, parentIdsJ, parentIdsArray, JNI_ABORT);
    return idGrouper;
}

bool isIndexIVFPQL2(faiss::Index * index) {
    faiss::Index * candidateIndex = index;
    // Unwrap the index if it is wrapped in IndexIDMap. Dynamic cast will "Safely converts pointers and references to
    // classes up, down, and sideways along the inheritance hierarchy." It will return a nullptr if the
    // cast fails. (ref: https://en.cppreference.com/w/cpp/language/dynamic_cast)
    if (auto indexIDMap = dynamic_cast<faiss::IndexIDMap *>(index)) {
        candidateIndex = indexIDMap->index;
    }

    // Check if the index is of type IndexIVFPQ. If so, confirm its metric type is
    // l2.
    if (auto indexIVFPQ = dynamic_cast<faiss::IndexIVFPQ *>(candidateIndex)) {
        return faiss::METRIC_L2 == indexIVFPQ->metric_type;
    }

    return false;
}

faiss::IndexIVFPQ * extractIVFPQIndex(faiss::Index * index) {
    faiss::Index * candidateIndex = index;
    if (auto indexIDMap = dynamic_cast<faiss::IndexIDMap *>(index)) {
        candidateIndex = indexIDMap->index;
    }

    faiss::IndexIVFPQ * indexIVFPQ;
    if ((indexIVFPQ = dynamic_cast<faiss::IndexIVFPQ *>(candidateIndex))) {
        return indexIVFPQ;
    }

    throw std::runtime_error("Unable to extract IVFPQ index. IVFPQ index not present.");
}

jobjectArray knn_jni::faiss_wrapper::RangeSearch(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexPointerJ,
                                                 jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ, jintArray parentIdsJ) {
    return knn_jni::faiss_wrapper::RangeSearchWithFilter(jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, nullptr, 0, parentIdsJ);
}

jobjectArray knn_jni::faiss_wrapper::RangeSearchWithFilter(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong indexPointerJ,
                                                           jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ, jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ) {
    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to indexReader");
    }

    float *rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    // The res will be freed by ~RangeSearchResult() in FAISS
    // The second parameter is always true, as lims is allocated by FAISS
    faiss::RangeSearchResult res(1, true);

    if(filterIdsJ != nullptr) {
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        std::unique_ptr<faiss::IDSelector> idSelector;
        if(filterIdsTypeJ == BITMAP) {
            idSelector.reset(new faiss::IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
        } else {
            faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
            idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
        }
        faiss::SearchParameters *searchParameters;
        faiss::SearchParametersHNSW hnswParams;
        faiss::SearchParametersIVF ivfParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader) {
            // Query param ef_search supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            hnswParams.sel = idSelector.get();
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        } else {
            auto ivfReader = dynamic_cast<const faiss::IndexIVF*>(indexReader->index);
            auto ivfFlatReader = dynamic_cast<const faiss::IndexIVFFlat*>(indexReader->index);
            if(ivfReader || ivfFlatReader) {
                ivfParams.sel = idSelector.get();
                searchParameters = &ivfParams;
            }
        }
        try {
            indexReader->range_search(1, rawQueryVector, radiusJ, &res, searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
    } else {
        faiss::SearchParameters *searchParameters = nullptr;
        faiss::SearchParametersHNSW hnswParams;
        std::unique_ptr<faiss::IDGrouperBitmap> idGrouper;
        std::vector<uint64_t> idGrouperBitmap;
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader!= nullptr) {
            // Query param ef_search supersedes ef_search provided during index setting.
            hnswParams.efSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, hnswReader->hnsw.efSearch);
            if (parentIdsJ != nullptr) {
                idGrouper = buildIDGrouperBitmap(jniUtil, env, parentIdsJ, &idGrouperBitmap);
                hnswParams.grp = idGrouper.get();
            }
            searchParameters = &hnswParams;
        }
        try {
            indexReader->range_search(1, rawQueryVector, radiusJ, &res, searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            throw;
        }
    }

    // lims is structured to support batched queries, it has a length of nq + 1 (where nq is the number of queries),
    // lims[i] - lims[i-1] gives the number of results for the i-th query. With a single query we used in k-NN,
    // res.lims[0] is always 0, and res.lims[1] gives the total number of matching entries found.
    int resultSize = res.lims[1];

    // Limit the result size to maxResultWindowJ so that we don't return more than the max result window
    // TODO: In the future, we should prevent this via FAISS's ResultHandler.
    if (resultSize > maxResultWindowJ) {
        resultSize = maxResultWindowJ;
    }

    jclass resultClass = jniUtil->FindClass(env,"org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);

    jobject result;
    for(int i = 0; i < resultSize; ++i) {
        result = jniUtil->NewObject(env, resultClass, allArgs, res.labels[i], res.distances[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
    }

    return results;
}
