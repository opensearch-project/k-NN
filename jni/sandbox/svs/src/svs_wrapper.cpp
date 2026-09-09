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
 * Native implementation of the isolated SVS engine (libopensearchknn_svs), on unmodified upstream faiss APIs.
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
#include "faiss/svs/IndexSVSVamanaLeanVec.h"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <jni.h>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

enum FilterIdsSelectorType{
    BITMAP = 0, BATCH = 1,
};

namespace {

// Filter as a Lucene FixedBitSet (jlong[] bits).
struct IDSelectorJlongBitmap : faiss::IDSelector {
    size_t n;
    const jlong* bitmap;

    IDSelectorJlongBitmap(size_t _n, const jlong* _bitmap)
      : faiss::IDSelector(),
        n(_n),
        bitmap(_bitmap) {
    }

    bool is_member(faiss::idx_t id) const final {
        const uint64_t index = id;
        const uint64_t i = index >> 6ULL;
        if (i >= n) {
            return false;
        }
        return (bitmap[i] >> (index & 63ULL)) & 1ULL;
    }
};

// Translates inner (sequential) ids through IndexIDMap#id_map before consulting the wrapped selector.
struct IDSelectorSvsTranslated : faiss::IDSelector {
    const std::vector<faiss::idx_t>& id_map;
    const faiss::IDSelector* sel;

    IDSelectorSvsTranslated(const std::vector<faiss::idx_t>& _id_map, const faiss::IDSelector* _sel)
      : id_map(_id_map),
        sel(_sel) {
    }

    bool is_member(faiss::idx_t id) const final {
        return sel->is_member(id_map[id]);
    }
};

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

// Applies the build parameters the factory description cannot carry.
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
    if (svsIndex->search_buffer_capacity < svsIndex->search_window_size) {
        svsIndex->search_buffer_capacity = svsIndex->search_window_size;
    }
    // java/lang/Double is not in JNIUtil's cached-class set; resolve it through the raw env.
    if ((value = parametersCpp.find(knn_jni::ALPHA)) != parametersCpp.end()) {
        jclass doubleClass = env->FindClass("java/lang/Double");
        jniUtil->HasExceptionInStack(env, "Could not find class java/lang/Double");
        if (doubleClass == nullptr || !env->IsInstanceOf(value->second, doubleClass)) {
            if (doubleClass != nullptr) {
                env->DeleteLocalRef(doubleClass);
            }
            throw std::runtime_error("alpha must be a floating-point value");
        }
        jmethodID doubleValue = env->GetMethodID(doubleClass, "doubleValue", "()D");
        jniUtil->HasExceptionInStack(env, "Could not find method doubleValue on java/lang/Double");
        svsIndex->alpha = static_cast<float>(env->CallDoubleMethod(value->second, doubleValue));
        jniUtil->HasExceptionInStack(env, "Could not call \"doubleValue\" method on Double");
        env->DeleteLocalRef(doubleClass);
    }
}

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

// Local PRNG: std::mt19937_64 leaks a default-visibility libstdc++ symbol from this otherwise symbol-clean library.
struct SplitMix64 {
    uint64_t state;
    uint64_t operator()() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

// Build handle returned by InitIndex and consumed by InsertToIndex/WriteIndex (which frees it). The graph is
// built in one pass at write time, so the whole (id, vector) stream is buffered here; trainTarget > 0 means a
// LeanVec projection is trained first on a uniform sample of the buffer.
struct SvsBuildContext {
    std::unique_ptr<faiss::IndexIDMap> idMap;
    int dim = 0;
    size_t trainTarget = 0;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    SplitMix64 rng{0x53565352u};

    void appendBatch(const faiss::idx_t* batchIds, size_t n, const float* batchVectors) {
        ids.insert(ids.end(), batchIds, batchIds + n);
        vectors.insert(vectors.end(), batchVectors, batchVectors + n * static_cast<size_t>(dim));
    }

    void buildAndAdd() {
        if (ids.empty()) {
            throw std::runtime_error("SVS static build reached the write phase with no vectors");
        }
        const size_t d = static_cast<size_t>(dim);
        if (trainTarget != 0) {
            size_t trainCount = std::min(trainTarget, ids.size());
            // Uniform sample (partial Fisher-Yates), never the first N: the head of a merged stream is skewed.
            std::vector<size_t> rows(ids.size());
            std::iota(rows.begin(), rows.end(), static_cast<size_t>(0));
            for (size_t i = 0; i < trainCount; ++i) {
                std::swap(rows[i], rows[i + rng() % (rows.size() - i)]);
            }
            std::vector<float> sample(trainCount * d);
            for (size_t i = 0; i < trainCount; ++i) {
                std::copy_n(vectors.data() + rows[i] * d, d, sample.data() + i * d);
            }
            idMap->train(static_cast<faiss::idx_t>(trainCount), sample.data());
            trainTarget = 0;
        }
        idMap->add_with_ids(static_cast<faiss::idx_t>(ids.size()), vectors.data(), ids.data());
        std::vector<float>().swap(vectors);
        std::vector<faiss::idx_t>().swap(ids);
    }
};

// Reads an integer from the encoder sub-map; fallback when absent or not positive (0 = default).
int64_t readEncoderIntParam(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env,
                            const std::unordered_map<std::string, jobject>& methodParams,
                            const std::string& paramName, int64_t fallback) {
    auto encoderIt = methodParams.find(knn_jni::ENCODER);
    if (encoderIt == methodParams.end() || encoderIt->second == nullptr) {
        return fallback;
    }
    auto encoderMap = jniUtil->ConvertJavaMapToCppMap(env, encoderIt->second);
    auto paramsIt = encoderMap.find(knn_jni::PARAMETERS);
    if (paramsIt == encoderMap.end() || paramsIt->second == nullptr) {
        return fallback;
    }
    auto encoderParams = jniUtil->ConvertJavaMapToCppMap(env, paramsIt->second);
    auto valueIt = encoderParams.find(paramName);
    if (valueIt == encoderParams.end() || valueIt->second == nullptr) {
        return fallback;
    }
    int64_t value = jniUtil->ConvertJavaObjectToCppInteger(env, valueIt->second);
    return value > 0 ? value : fallback;
}

// "SVSVamana64,LeanVec4x8_192" -> "SVSVamana64,LVQ4x8" for segments below the rough training threshold;
// LeanVec8x8 has no LVQ8x8 counterpart and degrades to LVQ4x8.
std::string rewriteLeanVecToLvq(const std::string& description) {
    size_t tokenStart = description.find(",LeanVec");
    if (tokenStart == std::string::npos) {
        throw std::runtime_error("Not a LeanVec index description: " + description);
    }
    size_t bitsStart = tokenStart + 8;
    size_t tokenEnd = description.find(',', bitsStart);
    std::string bits = description.substr(bitsStart, (tokenEnd == std::string::npos ? description.size() : tokenEnd) - bitsStart);
    size_t dimsSuffix = bits.find('_');
    if (dimsSuffix != std::string::npos) {
        bits = bits.substr(0, dimsSuffix);
    }
    if (bits == "8x8") {
        bits = "4x8";
    }
    if (bits != "4x4" && bits != "4x8") {
        throw std::runtime_error("Unsupported LeanVec kind for LVQ fallback in description: " + description);
    }
    std::string rewritten = description.substr(0, tokenStart) + ",LVQ" + bits;
    if (tokenEnd != std::string::npos) {
        rewritten += description.substr(tokenEnd);
    }
    return rewritten;
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

    if (threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // LeanVec ladder: below the rough threshold build the LVQ equivalent, else train at write time.
    size_t trainTarget = 0;
    if (indexDescriptionCpp.find(",LeanVec") != std::string::npos) {
        int64_t roughThreshold = readEncoderIntParam(
            jniUtil, env, subParametersCpp, knn_jni::LEANVEC_ROUGH_TRAINING_THRESHOLD,
            knn_jni::LEANVEC_DEFAULT_ROUGH_TRAINING_THRESHOLD);
        int64_t finalThreshold = readEncoderIntParam(
            jniUtil, env, subParametersCpp, knn_jni::LEANVEC_TRAINING_THRESHOLD,
            knn_jni::LEANVEC_DEFAULT_TRAINING_THRESHOLD);
        if (numDocs < roughThreshold) {
            indexDescriptionCpp = rewriteLeanVecToLvq(indexDescriptionCpp);
        } else {
            trainTarget = static_cast<size_t>(std::min<int64_t>(finalThreshold, numDocs));
        }
    }

    std::unique_ptr<faiss::Index> index(faiss::index_factory(static_cast<int>(dimJ), indexDescriptionCpp.c_str(), metric));

    auto svsIndex = dynamic_cast<faiss::IndexSVSVamana*>(index.get());
    if (svsIndex == nullptr) {
        throw std::runtime_error("Index description \"" + indexDescriptionCpp + "\" is not an SVS Vamana index");
    }
    applySVSVamanaParameters(jniUtil, env, subParametersCpp, svsIndex);

    // Segments are immutable: build the graph in one pass over the complete segment.
    svsIndex->is_static = true;
    // Nothing calls reconstruct(); without this faiss keeps a full FP32 copy of every vector.
    svsIndex->stored_vectors_valid = false;

    if (!index->is_trained && trainTarget == 0) {
        throw std::runtime_error("Index is not trained");
    }

    auto idMap = std::make_unique<faiss::IndexIDMap>(index.get());
    idMap->own_fields = true;
    index.release();

    auto context = std::make_unique<SvsBuildContext>();
    context->dim = static_cast<int>(dimJ);
    if (!idMap->is_trained && trainTarget > 0) {
        context->trainTarget = trainTarget;
    }
    if (numDocs > 0) {
        context->ids.reserve(static_cast<size_t>(numDocs));
        context->vectors.reserve(static_cast<size_t>(numDocs) * static_cast<size_t>(dimJ));
    }
    context->idMap = std::move(idMap);

    return reinterpret_cast<jlong>(context.release());
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

    auto *context = reinterpret_cast<SvsBuildContext *>(indexAddressJ);
    extractSVSVamana(context->idMap.get());

    // The off-heap vector block is core-owned and reused per batch, so the buffer must copy.
    context->appendBatch(ids.data(), static_cast<size_t>(numVectors), inputVectors->data());
}

void knn_jni::svs_wrapper::WriteIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject output,
                                      jlong indexAddressJ) {
    if (output == nullptr) {
        throw std::runtime_error("Index output stream cannot be null");
    }

    knn_jni::stream::NativeEngineIndexOutputMediator mediator {jniUtil, env, output};
    knn_jni::stream::FaissOpenSearchIOWriter writer {&mediator};

    std::unique_ptr<SvsBuildContext> context(reinterpret_cast<SvsBuildContext *>(indexAddressJ));

    context->buildAndAdd();

    try {
        faiss::write_index(context->idMap.get(), &writer);
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
    // Query-time values supersede the index-level defaults; SVS requires capacity >= window.
    svsVamanaParams.search_window_size = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::SEARCH_WINDOW_SIZE, svsVamanaReader->search_window_size);
    svsVamanaParams.search_buffer_capacity = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::SEARCH_BUFFER_CAPACITY, svsVamanaReader->search_buffer_capacity);
    svsVamanaParams.search_buffer_capacity = std::max(
        static_cast<size_t>(svsVamanaParams.search_buffer_capacity), svsVamanaParams.search_window_size);

    // The SVS runtime does not pad short results with faiss's -1 sentinel: clamp k and pre-fill the sentinels.
    int k = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(kJ), svsVamanaReader->ntotal));
    if (k <= 0) {
        std::vector<faiss::idx_t> emptyIds;
        std::vector<float> emptyDis;
        return buildQueryResults(jniUtil, env, emptyIds, emptyDis, 0);
    }
    std::vector<float> dis(k, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> ids(k, -1);

    omp_set_num_threads(1);

    // Pin order: everything that can throw runs before a pin or inside the try that owns it.
    if (filterIdsJ != nullptr) {
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        float* rawQueryVector = nullptr;
        try {
            std::unique_ptr<faiss::IDSelector> idSelector;
            if (filterIdsTypeJ == BITMAP) {
                idSelector.reset(new IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
            } else {
                faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
                idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
            }
            svsVamanaParams.sel = idSelector.get();
            rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
            indexReader->search(1, rawQueryVector, k, dis.data(), ids.data(), &svsVamanaParams);
        } catch (...) {
            if (rawQueryVector != nullptr) {
                jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            }
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
    } else {
        float* rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
        try {
            indexReader->search(1, rawQueryVector, k, dis.data(), ids.data(), &svsVamanaParams);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
    }

    return buildQueryResults(jniUtil, env, ids, dis, k);
}

jobjectArray knn_jni::svs_wrapper::RangeSearch(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                               jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ,
                                               jint maxResultWindowJ) {
    return RangeSearch_WithFilter(jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ,
                                  maxResultWindowJ, nullptr, 0);
}

jobjectArray knn_jni::svs_wrapper::RangeSearch_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env,
                                                          jlong indexPointerJ, jfloatArray queryVectorJ,
                                                          jfloat radiusJ, jobject methodParamsJ,
                                                          jint maxResultWindowJ, jlongArray filterIdsJ,
                                                          jint filterIdsTypeJ) {
    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }
    if (radiusJ <= 0) {
        throw std::runtime_error("SVS radial search requires a strictly positive radius");
    }

    auto *indexReader = reinterpret_cast<faiss::IndexIDMap *>(indexPointerJ);
    auto *svsVamanaReader = extractSVSVamana(indexReader);

    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    faiss::SearchParametersSVSVamana svsVamanaParams;
    svsVamanaParams.search_window_size = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::SEARCH_WINDOW_SIZE, svsVamanaReader->search_window_size);
    svsVamanaParams.search_buffer_capacity = knn_jni::commons::getIntegerMethodParameter(
        env, jniUtil, methodParams, knn_jni::SEARCH_BUFFER_CAPACITY, svsVamanaReader->search_buffer_capacity);
    svsVamanaParams.search_buffer_capacity = std::max(
        static_cast<size_t>(svsVamanaParams.search_buffer_capacity), svsVamanaParams.search_window_size);

    // IndexIDMap::range_search rebuilds a plain SearchParameters and drops the SVS fields, so call the inner
    // index directly and translate ids through id_map both ways.
    faiss::RangeSearchResult res(1, true);

    omp_set_num_threads(1);

    if (filterIdsJ != nullptr) {
        int filterIdsLength = jniUtil->GetJavaLongArrayLength(env, filterIdsJ);
        jlong *filteredIdsArray = jniUtil->GetLongArrayElements(env, filterIdsJ, nullptr);
        float* rawQueryVector = nullptr;
        try {
            std::unique_ptr<faiss::IDSelector> idSelector;
            if (filterIdsTypeJ == BITMAP) {
                idSelector.reset(new IDSelectorJlongBitmap(filterIdsLength, filteredIdsArray));
            } else {
                faiss::idx_t* batchIndices = reinterpret_cast<faiss::idx_t*>(filteredIdsArray);
                idSelector.reset(new faiss::IDSelectorBatch(filterIdsLength, batchIndices));
            }
            IDSelectorSvsTranslated translatedSelector(indexReader->id_map, idSelector.get());
            svsVamanaParams.sel = &translatedSelector;
            rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
            svsVamanaReader->range_search(1, rawQueryVector, radiusJ, &res, &svsVamanaParams);
        } catch (...) {
            if (rawQueryVector != nullptr) {
                jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            }
            jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
        jniUtil->ReleaseLongArrayElements(env, filterIdsJ, filteredIdsArray, JNI_ABORT);
    } else {
        float* rawQueryVector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr);
        try {
            svsVamanaReader->range_search(1, rawQueryVector, radiusJ, &res, &svsVamanaParams);
        } catch (...) {
            jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
            throw;
        }
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryVector, JNI_ABORT);
    }

    int resultSize = static_cast<int>(std::min<size_t>(res.lims[1], static_cast<size_t>(maxResultWindowJ)));

    jclass resultClass = jniUtil->FindClass(env, "org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);
    for (int i = 0; i < resultSize; ++i) {
        faiss::idx_t label = indexReader->id_map[res.labels[i]];
        jobject result = jniUtil->NewObject(env, resultClass, allArgs, label, res.distances[i]);
        jniUtil->SetObjectArrayElement(env, results, i, result);
        env->DeleteLocalRef(result);
    }
    return results;
}

void knn_jni::svs_wrapper::Free(jlong indexPointerJ) {
    auto *index = reinterpret_cast<faiss::Index*>(indexPointerJ);
    delete index;
}

void knn_jni::svs_wrapper::InitLibrary() {
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
    // Cosine vectors are normalized at the Java layer.
    if (spaceType == knn_jni::COSINESIMIL) {
        return faiss::METRIC_INNER_PRODUCT;
    }
    throw std::runtime_error("Invalid spaceType: " + spaceType);
}
