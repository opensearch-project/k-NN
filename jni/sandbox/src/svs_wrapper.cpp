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

/*
 * Native implementation for the isolated SVS engine (libopensearchknn_svs). Deliberately minimal and
 * self-contained: it only knows how to build, write, load, query (top-k, optionally pre-filtered) and free
 * an SVS Vamana index, using nothing beyond unmodified upstream faiss APIs (index_factory, IndexIDMap,
 * read_index/write_index, IDSelector) plus the SVS surface (IndexSVSVamana, SearchParametersSVSVamana).
 * It shares no translation units with the main faiss JNI library other than the generic JNI marshalling
 * helpers (jni_util/commons), which are faiss-free.
 */

#include "svs_wrapper.h"

#include "jni_util.h"
#include "svs_constants.h"
#include "commons.h"
#include "faiss_stream_support.h"

#include "faiss/Index.h"
#include "faiss/IndexIDMap.h"
#include "faiss/impl/IDSelector.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/svs/IndexSVSVamana.h"

#include <omp.h>

#include <algorithm>
#include <jni.h>
#include <memory>
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
    IDSelectorJlongBitmap(size_t _n, const jlong* _bitmap)
      : IDSelector(),
        n(_n),
        bitmap(_bitmap) {
    }

    bool is_member(idx_t id) const final {
        const uint64_t index = id;
        const uint64_t i = index >> 6ULL;  // div 64
        if (i >= n) {
            return false;
        }
        return (bitmap[i] >> (index & 63ULL)) & 1ULL;
    }
};  // class IDSelectorJlongBitmap

}  // namespace faiss

namespace {

// Extracts the IndexSVSVamana out of the IndexIDMap wrapper, throwing if the index is anything else.
// This library only ever serves .svs files containing SVS Vamana indices.
faiss::IndexSVSVamana* extractSVSVamana(faiss::IndexIDMap* idMap) {
    if (idMap == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }
    auto svsIndex = dynamic_cast<faiss::IndexSVSVamana*>(idMap->index);
    if (svsIndex == nullptr) {
        throw std::runtime_error("Index is not an SVS Vamana index");
    }
    return svsIndex;
}

// Applies the svs_vamana build parameters that the index factory description cannot carry. The
// search_window_size / search_buffer_capacity values become the index-level defaults that query-time
// method_parameters may override.
void applySVSVamanaParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                              const std::unordered_map<std::string, jobject>& parametersCpp,
                              faiss::IndexSVSVamana* svsIndex) {
    std::unordered_map<std::string, jobject>::const_iterator value;
    if ((value = parametersCpp.find(knn_jni::CONSTRUCTION_WINDOW_SIZE)) != parametersCpp.end()) {
        svsIndex->construction_window_size = static_cast<size_t>(jniUtil->ConvertJavaObjectToCppInteger(env, value->second));
    }
    if ((value = parametersCpp.find(knn_jni::SEARCH_WINDOW_SIZE)) != parametersCpp.end()) {
        svsIndex->search_window_size = static_cast<size_t>(jniUtil->ConvertJavaObjectToCppInteger(env, value->second));
    }
    if ((value = parametersCpp.find(knn_jni::SEARCH_BUFFER_CAPACITY)) != parametersCpp.end()) {
        svsIndex->search_buffer_capacity = static_cast<size_t>(jniUtil->ConvertJavaObjectToCppInteger(env, value->second));
    }
    // SVS requires search_buffer_capacity >= search_window_size.
    if (svsIndex->search_buffer_capacity < svsIndex->search_window_size) {
        svsIndex->search_buffer_capacity = svsIndex->search_window_size;
    }
    // Only override alpha when present: unset keeps the SVS metric-dependent default (1.2 L2, 0.95 IP).
    // java/lang/Double is not in JNIUtil's cached-class set, so resolve it through the raw JNI env.
    if ((value = parametersCpp.find(knn_jni::ALPHA)) != parametersCpp.end()) {
        jclass doubleClass = env->FindClass("java/lang/Double");
        jniUtil->HasExceptionInStack(env, "Could not find class java/lang/Double");
        if (doubleClass != nullptr && env->IsInstanceOf(value->second, doubleClass)) {
            jmethodID doubleValue = env->GetMethodID(doubleClass, "doubleValue", "()D");
            jniUtil->HasExceptionInStack(env, "Could not find method doubleValue on java/lang/Double");
            svsIndex->alpha = static_cast<float>(env->CallDoubleMethod(value->second, doubleValue));
            jniUtil->HasExceptionInStack(env, "Could not call \"doubleValue\" method on Double");
        } else {
            // Fall back for integral JSON values (e.g. alpha: 2); Integer IS in the cached set.
            svsIndex->alpha = static_cast<float>(jniUtil->ConvertJavaObjectToCppInteger(env, value->second));
        }
        if (doubleClass != nullptr) {
            env->DeleteLocalRef(doubleClass);
        }
    }
}

// Builds the KNNQueryResult[] from raw search output; results shorter than k are marked with id -1.
jobjectArray buildQueryResults(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env,
                               const std::vector<faiss::idx_t>& ids, const std::vector<float>& dis, int k) {
    int resultSize = k;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }

    jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
    for (int i = 0; i < resultSize; ++i) {
        jobject result = jniUtil->NewObject(env, resultClass, allArgs, ids[i], dis[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
        env->DeleteLocalRef(result);
    }
    return results;
}

}  // namespace

jlong knn_jni::svs_wrapper::InitIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong numDocs, jint dimJ,
                                      jobject parametersJ) {
    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);
    jniUtil->DeleteLocalRef(env, spaceTypeJ);

    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::INDEX_DESCRIPTION);
    std::string indexDescriptionCpp(jniUtil->ConvertJavaObjectToCppString(env, indexDescriptionJ));
    jniUtil->DeleteLocalRef(env, indexDescriptionJ);

    int threadCount = 0;
    if (parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        threadCount = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
    }

    std::unordered_map<std::string, jobject> subParametersCpp;
    if (parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersCpp[knn_jni::PARAMETERS]);
    }

    // Setting omp threads affects only the current thread's parallel regions.
    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // The description (e.g. "SVSVamana64,LVQ4x4") comes from the sandbox method/encoder mapping and is
    // parsed by the unmodified upstream faiss factory.
    std::unique_ptr<faiss::Index> index(faiss::index_factory(static_cast<int>(dimJ), indexDescriptionCpp.c_str(), metric));

    auto svsIndex = dynamic_cast<faiss::IndexSVSVamana*>(index.get());
    if (svsIndex == nullptr) {
        throw std::runtime_error("Index description \"" + indexDescriptionCpp + "\" is not an SVS Vamana index");
    }
    applySVSVamanaParameters(jniUtil, env, subParametersCpp, svsIndex);

    if (!index->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    auto idMap = std::make_unique<faiss::IndexIDMap>(index.get());
    // The IDMap must free the inner index when it is itself freed.
    idMap->own_fields = true;
    index.release();

    return reinterpret_cast<jlong>(idMap.release());
}

void knn_jni::svs_wrapper::InsertToIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                         jlong vectorsAddressJ, jint dimJ, jlong indexAddressJ, jint threadCount) {
    if (idsJ == nullptr) {
        throw std::runtime_error("IDs cannot be null");
    }
    if (vectorsAddressJ <= 0) {
        throw std::runtime_error("VectorsAddress cannot be less than 0");
    }
    if (dimJ <= 0) {
        throw std::runtime_error("Vectors dimensions cannot be less than or equal to 0");
    }

    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
    int dim = static_cast<int>(dimJ);
    int numVectors = static_cast<int>(inputVectors->size() / static_cast<uint64_t>(dim));
    if (numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }
    auto ids = jniUtil->ConvertJavaIntArrayToCppIntVector(env, idsJ);

    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    auto *idMap = reinterpret_cast<faiss::IndexIDMap *>(indexAddressJ);
    extractSVSVamana(idMap);  // validate the pointer really is an SVS index
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void knn_jni::svs_wrapper::WriteIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject output,
                                      jlong indexAddressJ) {
    if (output == nullptr) {
        throw std::runtime_error("Index output stream cannot be null");
    }

    knn_jni::stream::NativeEngineIndexOutputMediator mediator {jniUtil, env, output};
    knn_jni::stream::FaissOpenSearchIOWriter writer {&mediator};

    // The index is freed after writing (the build strategy creates it solely to write it).
    std::unique_ptr<faiss::IndexIDMap> idMap(reinterpret_cast<faiss::IndexIDMap *>(indexAddressJ));
    try {
        faiss::write_index(idMap.get(), &writer);
        writer.flush();
    } catch (std::exception &e) {
        throw std::runtime_error(std::string("Failed to write index to disk, error=") + e.what());
    }
}

jlong knn_jni::svs_wrapper::LoadIndexWithStream(faiss::IOReader* ioReader) {
    if (ioReader == nullptr) {
        throw std::runtime_error("IOReader cannot be null");
    }

    std::unique_ptr<faiss::Index> indexReader(faiss::read_index(ioReader, faiss::IO_FLAG_READ_ONLY));

    // .svs files only ever contain IDMap(IndexSVSVamana); refuse anything else outright.
    auto idMap = dynamic_cast<faiss::IndexIDMap*>(indexReader.get());
    if (idMap == nullptr || dynamic_cast<faiss::IndexSVSVamana*>(idMap->index) == nullptr) {
        throw std::runtime_error("Loaded index is not an SVS Vamana index");
    }

    return reinterpret_cast<jlong>(indexReader.release());
}

jobjectArray knn_jni::svs_wrapper::QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                              jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ) {
    return QueryIndex_WithFilter(jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, nullptr, 0);
}

jobjectArray knn_jni::svs_wrapper::QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env,
                                                         jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ,
                                                         jobject methodParamsJ, jlongArray filterIdsJ,
                                                         jint filterIdsTypeJ) {
    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);
    auto *svsVamanaReader = extractSVSVamana(indexReader);

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    faiss::SearchParametersSVSVamana svsVamanaParams;
    // Query-time search_window_size supersedes the index-level value; buffer capacity tracks it when the
    // index-level capacity is smaller (SVS requires capacity >= window).
    svsVamanaParams.search_window_size = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::SEARCH_WINDOW_SIZE, svsVamanaReader->search_window_size);
    svsVamanaParams.search_buffer_capacity = std::max(
        static_cast<size_t>(svsVamanaReader->search_buffer_capacity), svsVamanaParams.search_window_size);

    // The SVS runtime does not pad missing results with faiss's -1 sentinel; slots beyond the number of
    // results found would otherwise surface as label 0 / distance 0. Clamp k to the segment's vector count
    // and pre-fill with the faiss sentinels so short result sets (e.g. under a restrictive pre-filter) are
    // detected by the trim below — which in turn lets the Java layer fall back to exact search.
    int k = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(kJ), svsVamanaReader->ntotal));
    if (k <= 0) {
        std::vector<faiss::idx_t> emptyIds;
        std::vector<float> emptyDis;
        return buildQueryResults(jniUtil, env, emptyIds, emptyDis, 0);
    }
    std::vector<float> dis(k, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> ids(k, -1);
    float* rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);

    // Set omp threads to 1 so no new OMP threads are created under the search threadpool.
    omp_set_num_threads(1);

    if (filterIdsJ != nullptr) {
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        std::unique_ptr<faiss::IDSelector> idSelector;
        if (filterIdsTypeJ == BITMAP) {
            idSelector.reset(new faiss::IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
        } else {
            faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
            idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
        }
        svsVamanaParams.sel = idSelector.get();
        try {
            indexReader->search(1, rawQueryVector, k, dis.data(), ids.data(), &svsVamanaParams);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
    } else {
        try {
            indexReader->search(1, rawQueryVector, k, dis.data(), ids.data(), &svsVamanaParams);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            throw;
        }
    }
    jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);

    return buildQueryResults(jniUtil, env, ids, dis, k);
}

void knn_jni::svs_wrapper::Free(jlong indexPointerJ) {
    auto *index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    delete index;
}

void knn_jni::svs_wrapper::InitLibrary() {
    // No global initialization required today; kept as the single hook for any future setup.
}

bool knn_jni::svs_wrapper::IsLvqLeanvecEnabled() {
    return faiss::IndexSVSVamana::is_lvq_leanvec_enabled();
}

faiss::MetricType knn_jni::svs_wrapper::TranslateSpaceToMetric(const std::string& spaceType) {
    if (spaceType == knn_jni::L2) {
        return faiss::METRIC_L2;
    }
    if (spaceType == knn_jni::INNER_PRODUCT) {
        return faiss::METRIC_INNER_PRODUCT;
    }
    // Vectors are normalized at the Java layer for cosine, so cosine is equivalent to inner product.
    if (spaceType == knn_jni::COSINESIMIL) {
        return faiss::METRIC_INNER_PRODUCT;
    }
    throw std::runtime_error("Invalid spaceType: " + spaceType);
}
