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
#include "knn_extension/faiss/MultiVectorResultCollectorFactory.h"

#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetaIndexes.h"
#include "faiss/Index.h"
#include "faiss/impl/IDSelector.h"

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

// Converts the int FilterIds to Faiss ids type array.
void convertFilterIdsToFaissIdType(const int* filterIds, int filterIdsLength, faiss::idx_t* convertedFilterIds);

// Concerts the FilterIds to BitMap
void buildFilterIdsBitMap(const int* filterIds, int filterIdsLength, uint8_t* bitsetVector);

os_faiss::MultiVectorResultCollectorFactory* buildResultCollectorFactory(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env, jintArray parentIdsJ);

void releaseResultCollectorFactory(os_faiss::MultiVectorResultCollectorFactory* collectorFactory);

void knn_jni::faiss_wrapper::CreateIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                         jobjectArray vectorsJ, jstring indexPathJ, jobject parametersJ) {

    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsJ == nullptr) {
        throw std::runtime_error("Vectors cannot be null");
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

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    // Read data set
    int numVectors = jniUtil->GetJavaObjectArrayLength(env, vectorsJ);
    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil->Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);

    // Create faiss index
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));

    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::index_factory(dim, indexDescriptionCpp.c_str(), metric));

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
    jniUtil->DeleteLocalRef(env, parametersJ);

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    auto idVector = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);
    faiss::IndexIDMap idMap = faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, dataset.data(), idVector.data());

    // Write the index to disk
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index(&idMap, indexPathCpp.c_str());
}

void knn_jni::faiss_wrapper::CreateIndexFromTemplate(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                                     jobjectArray vectorsJ, jstring indexPathJ,
                                                     jbyteArray templateIndexJ, jobject parametersJ) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }

    if (vectorsJ == nullptr) {
        throw std::runtime_error("Vectors cannot be null");
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
    int numVectors = jniUtil->GetJavaObjectArrayLength(env, vectorsJ);
    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil->Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);

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
    idMap.add_with_ids(numVectors, dataset.data(), idVector.data());

    // Write the index to disk
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index(&idMap, indexPathCpp.c_str());
}

jlong knn_jni::faiss_wrapper::LoadIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));
    faiss::Index* indexReader = faiss::read_index(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY);
    return (jlong) indexReader;
}

jobjectArray knn_jni::faiss_wrapper::QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                jfloatArray queryVectorJ, jint kJ, jintArray parentIdsJ) {
    return knn_jni::faiss_wrapper::QueryIndex_WithFilter(jniUtil, env, indexPointerJ, queryVectorJ, kJ, nullptr, 0, parentIdsJ);
}

jobjectArray knn_jni::faiss_wrapper::QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                jfloatArray queryVectorJ, jint kJ, jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
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
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader) {
            // Setting the ef_search value equal to what was provided during index creation. SearchParametersHNSW has a default
            // value of ef_search = 16 which will then be used.
            hnswParams.efSearch = hnswReader->hnsw.efSearch;
            hnswParams.sel = idSelector.get();
            hnswParams.col = buildResultCollectorFactory(jniUtil, env, parentIdsJ);
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
            indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            releaseResultCollectorFactory(dynamic_cast<os_faiss::MultiVectorResultCollectorFactory*>(hnswParams.col));
            throw;
        }
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
        releaseResultCollectorFactory(dynamic_cast<os_faiss::MultiVectorResultCollectorFactory*>(hnswParams.col));
    } else {
        faiss::SearchParameters *searchParameters = nullptr;
        faiss::SearchParametersHNSW hnswParams;
        auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
        if(hnswReader!= nullptr && parentIdsJ != nullptr) {
            // Setting the ef_search value equal to what was provided during index creation. SearchParametersHNSW has a default
            // value of ef_search = 16 which will then be used.
            hnswParams.efSearch = hnswReader->hnsw.efSearch;
            hnswParams.col = buildResultCollectorFactory(jniUtil, env, parentIdsJ);
            searchParameters = &hnswParams;
        }
        try {
            indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data(), searchParameters);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
            releaseResultCollectorFactory(dynamic_cast<os_faiss::MultiVectorResultCollectorFactory*>(hnswParams.col));
            throw;
        }
        releaseResultCollectorFactory(dynamic_cast<os_faiss::MultiVectorResultCollectorFactory*>(hnswParams.col));
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

void knn_jni::faiss_wrapper::Free(jlong indexPointer) {
    auto *indexWrapper = reinterpret_cast<faiss::Index*>(indexPointer);
    delete indexWrapper;
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

faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType) {
    if (spaceType == knn_jni::L2) {
        return faiss::METRIC_L2;
    }

    if (spaceType == knn_jni::INNER_PRODUCT) {
        return faiss::METRIC_INNER_PRODUCT;
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

os_faiss::MultiVectorResultCollectorFactory* buildResultCollectorFactory(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env, jintArray parentIdsJ) {
    if (parentIdsJ == nullptr) {
        return nullptr;
    }
    int *parentIdsArray = jniUtil->GetIntArrayElements(env, parentIdsJ, nullptr);
    int parentIdsLength = jniUtil->GetJavaIntArrayLength(env, parentIdsJ);
    auto* parent_id_filter = new FixedBitSet(parentIdsArray, parentIdsLength);
    jniUtil->ReleaseIntArrayElements(env, parentIdsJ, parentIdsArray, JNI_ABORT);
    return new os_faiss::MultiVectorResultCollectorFactory(parent_id_filter);
}

void releaseResultCollectorFactory(os_faiss::MultiVectorResultCollectorFactory* collectorFactory) {
    if (collectorFactory == nullptr) {
        return;
    }
    delete collectorFactory->parent_bit_set;
    delete collectorFactory;
}