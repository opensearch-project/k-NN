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
#include "nmslib_wrapper.h"

#include "commons.h"

#include "init.h"
#include "index.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"

#include <jni.h>
#include <string>

#include "hnswquery.h"


std::string TranslateSpaceType(const std::string& spaceType);

// We do not use label functionality of nmslib so we pass default label. Setting as a const allows us to avoid a few
// allocations
const similarity::LabelType DEFAULT_LABEL = -1;

void knn_jni::nmslib_wrapper::CreateIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                          jlong vectorsAddressJ, jint dimJ,
                                          jstring indexPathJ, jobject parametersJ) {

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

    // Handle parameters
    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);
    std::vector<std::string> indexParameters;

    // Algorithm parameters will be in a sub map
    if(parametersCpp.find(knn_jni::PARAMETERS) != parametersCpp.end()) {
        jobject subParametersJ = parametersCpp[knn_jni::PARAMETERS];
        auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, subParametersJ);

        if(subParametersCpp.find(knn_jni::EF_CONSTRUCTION) != subParametersCpp.end()) {
            auto efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, subParametersCpp[knn_jni::EF_CONSTRUCTION]);
            indexParameters.push_back(knn_jni::EF_CONSTRUCTION_NMSLIB + "=" + std::to_string(efConstruction));
        }

        if(subParametersCpp.find(knn_jni::M) != subParametersCpp.end()) {
            auto m = jniUtil->ConvertJavaObjectToCppInteger(env, subParametersCpp[knn_jni::M]);
            indexParameters.push_back(knn_jni::M_NMSLIB + "=" + std::to_string(m));
        }

        jniUtil->DeleteLocalRef(env, subParametersJ);
    }

    if(parametersCpp.find(knn_jni::INDEX_THREAD_QUANTITY) != parametersCpp.end()) {
        auto indexThreadQty = jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp[knn_jni::INDEX_THREAD_QUANTITY]);
        indexParameters.push_back(knn_jni::INDEX_THREAD_QUANTITY + "=" + std::to_string(indexThreadQty));
    }

    jniUtil->DeleteLocalRef(env, parametersJ);

    // Get the path to save the index
    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);

    std::unique_ptr<similarity::Space<float>> space;
    space.reset(similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceTypeCpp,similarity::AnyParams()));

    // Get number of ids and vectors and dimension
    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddressJ);
    int dim = (int)dimJ;
    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) ( inputVectors->size() / (uint64_t) dim);
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    int numIds = jniUtil->GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    // Read dataset
    similarity::ObjectVector dataset;
    dataset.reserve(numVectors);
    int* idsCpp;
    try {
        // Read in data set
        idsCpp = jniUtil->GetIntArrayElements(env, idsJ, nullptr);
        size_t vectorSizeInBytes = dim*sizeof(float);
        // vectorPointer needs to be unsigned long long, this will ensure that out of range doesn't happen for this pointer
        // when the values of numVectors * dim becomes very large.
        // Example: for 10M vectors of 1536 dim vectorPointer max value will be ~15.3B which is already > range of ints.
        // keeping it unsigned long long we will never go above the range.
        unsigned long long vectorPointer = 0;

        // Allocate a large buffer that will contain all the vectors. Allocating the objects in one large buffer as
        // opposed to individually will prevent heap fragmentation. We have observed that allocating individual
        // objects causes RSS to rise throughout the lifetime of a process
        // (see https://github.com/opensearch-project/k-NN/issues/772 and
        // https://github.com/opensearch-project/k-NN/issues/72). This is because, in typical systems, small
        // allocations will reside on some kind of heap managed by an allocator. Once freed, the allocator does not
        // always return the memory to the OS. If the heap gets fragmented, this will cause the allocator
        // to ask for more memory, causing RSS to grow. On large allocations (> 128 kb), most allocators will
        // internally use mmap. Once freed, unmap will be called, which will immediately return memory to the OS
        // which in turn prevents RSS from growing out of control. Wrap with a smart pointer so that buffer will be
        // freed once variable goes out of scope. For reference, the code that specifies the layout of the buffer can be
        // found: https://github.com/nmslib/nmslib/blob/v2.1.1/similarity_search/include/object.h#L61-L75
        std::unique_ptr<char[]> objectBuffer(new char[(similarity::ID_SIZE + similarity::LABEL_SIZE + similarity::DATALENGTH_SIZE + vectorSizeInBytes) * numVectors]);
        char* ptr = objectBuffer.get();
        for (int i = 0; i < numVectors; i++) {
            dataset.push_back(new similarity::Object(ptr));

            memcpy(ptr, &idsCpp[i], similarity::ID_SIZE);
            ptr += similarity::ID_SIZE;
            memcpy(ptr, &DEFAULT_LABEL, similarity::LABEL_SIZE);
            ptr += similarity::LABEL_SIZE;
            memcpy(ptr, &vectorSizeInBytes, similarity::DATALENGTH_SIZE);
            ptr += similarity::DATALENGTH_SIZE;

            memcpy(ptr, &(inputVectors->at(vectorPointer)), vectorSizeInBytes);
            ptr += vectorSizeInBytes;
            vectorPointer += dim;
        }
        jniUtil->ReleaseIntArrayElements(env, idsJ, idsCpp, JNI_ABORT);

        // Releasing the vectorsAddressJ memory as that is not required once we have created the index.
        // This is not the ideal approach, please refer this gh issue for long term solution:
        // https://github.com/opensearch-project/k-NN/issues/1600
        //commons::freeVectorData(vectorsAddressJ);
        delete inputVectors;

        std::unique_ptr<similarity::Index<float>> index;
        index.reset(similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceTypeCpp, *(space), dataset));
        index->CreateIndex(similarity::AnyParams(indexParameters));
        index->SaveIndex(indexPathCpp);

        for (auto & it : dataset) {
            delete it;
        }
    } catch (...) {
        for (auto & it : dataset) {
            delete it;
        }

        jniUtil->ReleaseIntArrayElements(env, idsJ, idsCpp, JNI_ABORT);
        throw;
    }
}

jlong knn_jni::nmslib_wrapper::LoadIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ,
                                         jobject parametersJ) {

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    std::string indexPathCpp(jniUtil->ConvertJavaStringToCppString(env, indexPathJ));

    auto parametersCpp = jniUtil->ConvertJavaMapToCppMap(env, parametersJ);

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(jniUtil->ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);

    // Parse query params
    std::vector<std::string> queryParams;

    if(parametersCpp.find("efSearch") != parametersCpp.end()) {
        auto efSearch = std::to_string(jniUtil->ConvertJavaObjectToCppInteger(env, parametersCpp["efSearch"]));
        queryParams.push_back("efSearch=" + efSearch);
    }

    // Load index
    knn_jni::nmslib_wrapper::IndexWrapper * indexWrapper;
    try {
        indexWrapper = new knn_jni::nmslib_wrapper::IndexWrapper(spaceTypeCpp);
        indexWrapper->index->LoadIndex(indexPathCpp);
        indexWrapper->index->SetQueryTimeParams(similarity::AnyParams(queryParams));
    } catch (...) {
        delete indexWrapper;
        throw;
    }

    return (jlong) indexWrapper;
}

jobjectArray knn_jni::nmslib_wrapper::QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                 jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    if (indexPointerJ == 0) {
        throw std::runtime_error("Invalid pointer to index");
    }

    auto *indexWrapper = reinterpret_cast<knn_jni::nmslib_wrapper::IndexWrapper*>(indexPointerJ);

    int dim	= jniUtil->GetJavaFloatArrayLength(env, queryVectorJ);

    float* rawQueryvector = jniUtil->GetFloatArrayElements(env, queryVectorJ, nullptr); // Have to call release on this

    std::unique_ptr<const similarity::Object> queryObject;
    try {
        queryObject.reset(new similarity::Object(-1, -1, dim*sizeof(float), rawQueryvector));
    } catch (...) {
        jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
        throw;
    }

    jniUtil->ReleaseFloatArrayElements(env, queryVectorJ, rawQueryvector, JNI_ABORT);
    std::unordered_map<std::string, jobject> methodParams;
    if (methodParamsJ != nullptr) {
        methodParams = jniUtil->ConvertJavaMapToCppMap(env, methodParamsJ);
    }

    int queryEfSearch = knn_jni::commons::getIntegerMethodParameter(env, jniUtil, methodParams, EF_SEARCH, -1);
    similarity::KNNQuery<float>* query; // TODO: Replace with smart pointers https://github.com/opensearch-project/k-NN/issues/1785
    std::unique_ptr<similarity::KNNQueue<float>> neighbors;
    try {
        if (queryEfSearch == -1) {
            query = new similarity::KNNQuery<float>(*(indexWrapper->space), queryObject.get(), kJ);
        } else {
            query = new similarity::HNSWQuery<float>(*(indexWrapper->space), queryObject.get(), kJ, queryEfSearch);
        }

        indexWrapper->index->Search(query);
        neighbors.reset(query->Result()->Clone());
    } catch (...) {
        if (query != nullptr) {
            delete query;
        } 
        throw;
    }
    delete query;
    
    int resultSize = neighbors->Size();
    jclass resultClass = jniUtil->FindClass(env,"org/opensearch/knn/index/query/KNNQueryResult");
    jmethodID allArgs = jniUtil->FindMethod(env, "org/opensearch/knn/index/query/KNNQueryResult", "<init>");

    jobjectArray results = jniUtil->NewObjectArray(env, resultSize, resultClass, nullptr);

    jobject result;
    float distance;
    long id;
    for(int i = 0; i < resultSize; ++i) {
        distance = neighbors->TopDistance();
        id = neighbors->Pop()->id();
        result = jniUtil->NewObject(env, resultClass, allArgs, id, distance);
        jniUtil->SetObjectArrayElement(env, results, i, result);
    }

    return results;
}

void knn_jni::nmslib_wrapper::Free(jlong indexPointerJ) {
    auto *indexWrapper = reinterpret_cast<knn_jni::nmslib_wrapper::IndexWrapper*>(indexPointerJ);
    delete indexWrapper;
}

void knn_jni::nmslib_wrapper::InitLibrary() {
    similarity::initLibrary();
}

std::string TranslateSpaceType(const std::string& spaceType) {
    if (spaceType == knn_jni::L2) {
        return spaceType;
    }

    if (spaceType == knn_jni::L1) {
        return spaceType;
    }

    if (spaceType == knn_jni::LINF) {
        return spaceType;
    }

    if (spaceType == knn_jni::COSINESIMIL) {
        return spaceType;
    }

    if (spaceType == knn_jni::INNER_PRODUCT) {
        return knn_jni::NEG_DOT_PRODUCT;
    }

    throw std::runtime_error("Invalid spaceType");
}
