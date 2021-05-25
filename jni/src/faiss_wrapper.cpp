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

#include "faiss/index_factory.h"
#include "faiss/MetaIndexes.h"
#include "faiss/index_io.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFFlat.h"

#include <jni.h>
#include <vector>
#include <string>
#include <faiss/impl/io.h>


// Translate space type to faiss metric
faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType);

// Set additional parameters on faiss index
void SetExtraParameters(JNIEnv *env, const std::unordered_map<std::string, jobject>& parametersCpp,
                        faiss::Index * index);

// Train an index with data provided
void InternalTrainIndex(faiss::Index * index, faiss::Index::idx_t n, const float* x);

void knn_jni::faiss_wrapper::CreateIndex(JNIEnv * env, jintArray idsJ, jobjectArray vectorsJ, jstring indexPathJ,
                                         jobject parametersJ) {

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
    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    // Read data set
    int numVectors = knn_jni::GetJavaObjectArrayLength(env, vectorsJ);
    int numIds = knn_jni::GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    int dim = knn_jni::GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = knn_jni::Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);

    // Create faiss index
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::METHOD);
    std::string indexDescriptionCpp(knn_jni::ConvertJavaObjectToCppString(env, indexDescriptionJ));

    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::index_factory(dim, indexDescriptionCpp.c_str(), metric));

    // Add extra parameters that cant be configured with the index factory
    if(parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        jobject subParametersJ = parametersCpp[knn_jni::PARAMETERS];
        auto subParametersCpp = knn_jni::ConvertJavaMapToCppMap(env, subParametersJ);
        SetExtraParameters(env, subParametersCpp, indexWriter.get());
        env->DeleteLocalRef(subParametersJ);
    }
    env->DeleteLocalRef(parametersJ);

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    auto idVector = knn_jni::ConvertJavaIntArrayToCppIntVector(env, idsJ);
    faiss::IndexIDMap idMap = faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, dataset.data(), idVector.data());

    // Write the index to disk
    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index(&idMap, indexPathCpp.c_str());
}

void knn_jni::faiss_wrapper::CreateIndexFromTemplate(JNIEnv * env, jintArray idsJ, jobjectArray vectorsJ,
                                                     jstring indexPathJ, jbyteArray templateIndexJ) {
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

    // Read data set
    int numVectors = knn_jni::GetJavaObjectArrayLength(env, vectorsJ);
    int numIds = knn_jni::GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    int dim = knn_jni::GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = knn_jni::Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);

    // Get vector of bytes from jbytearray
    int indexBytesCount = knn_jni::GetJavaBytesArrayLength(env, templateIndexJ);
    jbyte * indexBytesJ = env->GetByteArrayElements(templateIndexJ, nullptr);
    if (indexBytesJ == nullptr) {
        knn_jni::HasExceptionInStack(env);
        throw std::runtime_error("Unable able to get byte array for template index");
    }

    faiss::VectorIOReader vectorIoReader;
    for (int i = 0; i < indexBytesCount; i++) {
        vectorIoReader.data.push_back((uint8_t) indexBytesJ[i]);
    }
    env->ReleaseByteArrayElements(templateIndexJ, indexBytesJ, JNI_ABORT);

    // Create faiss index
    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::read_index(&vectorIoReader, 0));

    auto idVector = knn_jni::ConvertJavaIntArrayToCppIntVector(env, idsJ);
    faiss::IndexIDMap idMap =  faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, dataset.data(), idVector.data());

    // Write the index to disk
    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    faiss::write_index(&idMap, indexPathCpp.c_str());
}

jlong knn_jni::faiss_wrapper::LoadIndex(JNIEnv * env, jstring indexPathJ) {
    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    faiss::Index* indexReader = faiss::read_index(indexPathCpp.c_str(), faiss::IO_FLAG_READ_ONLY);
    return (jlong) indexReader;
}

jobjectArray knn_jni::faiss_wrapper::QueryIndex(JNIEnv * env, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexReader = reinterpret_cast<faiss::Index*>(indexPointerJ);

    if (indexReader == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    int dim	= knn_jni::GetJavaFloatArrayLength(env, queryVectorJ);
    std::vector<float> dis(kJ * dim);
    std::vector<faiss::Index::idx_t> ids(kJ * dim);
    float* rawQueryvector = env->GetFloatArrayElements(queryVectorJ, nullptr); // Have to call release on this
    if (rawQueryvector == nullptr) {
        knn_jni::HasExceptionInStack(env);
        throw std::runtime_error("Unable to get float elements from query vector");
    }

    try {
        indexReader->search(1, rawQueryvector, kJ, dis.data(), ids.data());
        knn_jni::HasExceptionInStack(env);
    } catch (...) {
        env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);
        knn_jni::HasExceptionInStack(env);
        throw;
    }
    env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);

    // If there are not k results, the results will be padded with -1. Find the first -1, and set result size to that
    // index
    int resultSize = kJ;
    auto it = std::find(ids.begin(), ids.end(), -1);
    if (it != ids.end()) {
        resultSize = it - ids.begin();
    }

    jclass resultClass = knn_jni::FindClass(env,"org/opensearch/knn/index/KNNQueryResult");
    jmethodID allArgs = knn_jni::FindMethod(env, resultClass, "<init>", "(IF)V");

    jobjectArray results = env->NewObjectArray(resultSize, resultClass, nullptr);
    knn_jni::HasExceptionInStack(env);
    if (results == nullptr) {
        throw std::runtime_error("Unable to allocate results array");
    }

    jobject result;
    for(int i = 0; i < resultSize; ++i) {
        result = env->NewObject(resultClass, allArgs, ids[i], dis[i]);
        if (result == nullptr) {
            knn_jni::HasExceptionInStack(env);
            throw std::runtime_error("Unable to create result");
        }
        env->SetObjectArrayElement(results, i, result);
        knn_jni::HasExceptionInStack(env);
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

jbyteArray knn_jni::faiss_wrapper::TrainIndex(JNIEnv * env, jobject parametersJ, jint dimensionJ,
                                              jlong trainVectorsPointerJ) {
    // First, we need to build the index
    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);

    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    faiss::MetricType metric = TranslateSpaceToMetric(spaceTypeCpp);

    // Create faiss index
    jobject indexDescriptionJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::METHOD);
    std::string indexDescriptionCpp(knn_jni::ConvertJavaObjectToCppString(env, indexDescriptionJ));

    std::unique_ptr<faiss::Index> indexWriter;
    indexWriter.reset(faiss::index_factory((int) dimensionJ, indexDescriptionCpp.c_str(), metric));

    // Add extra parameters that cant be configured with the index factory
    if(parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        jobject subParametersJ = parametersCpp[knn_jni::PARAMETERS];
        auto subParametersCpp = knn_jni::ConvertJavaMapToCppMap(env, subParametersJ);
        SetExtraParameters(env, subParametersCpp, indexWriter.get());
        env->DeleteLocalRef(subParametersJ);
    }

    // Train index if needed
    auto *trainingVectorsPointerCpp = reinterpret_cast<std::vector<float>*>(trainVectorsPointerJ);
    int numVectors = trainingVectorsPointerCpp->size()/(int) dimensionJ;
    if(!indexWriter->is_trained) {
        InternalTrainIndex(indexWriter.get(), numVectors, trainingVectorsPointerCpp->data());
    }
    env->DeleteLocalRef(parametersJ);

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

    jbyteArray ret = env->NewByteArray(vectorIoWriter.data.size());
    env->SetByteArrayRegion(ret, 0, vectorIoWriter.data.size(), jbytesBuffer.get());
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

void SetExtraParameters(JNIEnv *env, const std::unordered_map<std::string, jobject>& parametersCpp,
                        faiss::Index * index) {

    std::unordered_map<std::string,jobject>::const_iterator value;
    if (auto * indexIvf = dynamic_cast<faiss::IndexIVF*>(index)) {
        if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
            indexIvf->nprobe = knn_jni::ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                && indexIvf->quantizer != nullptr) {
            auto subParametersCpp = knn_jni::ConvertJavaMapToCppMap(env, value->second);
            SetExtraParameters(env, subParametersCpp, indexIvf->quantizer);
        }
    }

    if (auto * indexHnsw = dynamic_cast<faiss::IndexHNSW*>(index)) {

        if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
            indexHnsw->hnsw.efConstruction = knn_jni::ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
            indexHnsw->hnsw.efSearch = knn_jni::ConvertJavaObjectToCppInteger(env, value->second);
        }
    }
}

void InternalTrainIndex(faiss::Index * index, faiss::Index::idx_t n, const float* x) {
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
