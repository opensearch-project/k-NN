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

#include <jni.h>
#include <string>

#include "init.h"
#include "index.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"


struct IndexWrapper {
    explicit IndexWrapper(const string& spaceType) {
        // Index gets constructed with a reference to data (see above) but is otherwise unused
        similarity::ObjectVector data;
        space.reset(similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceType, similarity::AnyParams()));
        index.reset(similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceType, *space, data));
    }
    std::unique_ptr<similarity::Space<float>> space;
    std::unique_ptr<similarity::Index<float>> index;
};

std::string TranslateSpaceType(const std::string& spaceType);

void knn_jni::nmslib_wrapper::CreateIndex(JNIEnv * env, jintArray idsJ, jobjectArray vectorsJ, jstring indexPathJ,
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

    // Handle parameters
    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);
    std::vector<std::string> indexParameters;
    if(parametersCpp.find("ef_construction") != parametersCpp.end()) {
        auto efConstruction = knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["ef_construction"]);
        indexParameters.push_back("efConstruction=" + std::to_string(efConstruction));
    }

    if(parametersCpp.find("m") != parametersCpp.end()) {
        auto m = knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["m"]);
        indexParameters.push_back("M=" + std::to_string(m));
    }
    env->DeleteLocalRef(parametersJ);

    // Get the path to save the index
    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);

    std::unique_ptr<similarity::Space<float>> space;
    space.reset(similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceTypeCpp,similarity::AnyParams()));

    // Get number of ids and vectors and dimension
    int numVectors = knn_jni::GetJavaObjectArrayLength(env, vectorsJ);
    int numIds = knn_jni::GetJavaIntArrayLength(env, idsJ);
    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }
    int dim = knn_jni::GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);

    // Read dataset
    similarity::ObjectVector dataset;
    int* idsCpp;
    try {
        // Read in data set
        idsCpp = env->GetIntArrayElements(idsJ, nullptr);
        if (idsCpp == nullptr) {
            HasExceptionInStack(env);
            throw std::runtime_error("Unable to get ids array");
        }

        float* floatArrayCpp;
        jfloatArray floatArrayJ;
        for (int i = 0; i < numVectors; i++) {
            floatArrayJ = (jfloatArray)env->GetObjectArrayElement(vectorsJ, i);
            HasExceptionInStack(env);

            if (dim != env->GetArrayLength(floatArrayJ)) {
                throw std::runtime_error("Dimension of vectors is inconsistent");
            }

            floatArrayCpp = env->GetFloatArrayElements(floatArrayJ, nullptr);
            if (floatArrayCpp == nullptr) {
                throw std::runtime_error("Unable to read float array");
            }

            dataset.push_back(new similarity::Object(idsCpp[i], -1, dim*sizeof(float), floatArrayCpp));
            env->ReleaseFloatArrayElements(floatArrayJ, floatArrayCpp, JNI_ABORT);
            HasExceptionInStack(env);
        }
        env->ReleaseIntArrayElements(idsJ, idsCpp, JNI_ABORT);
        HasExceptionInStack(env);

        std::unique_ptr<similarity::Index<float>> index;
        index.reset(similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceTypeCpp, *(space), dataset));
        index->CreateIndex(similarity::AnyParams(indexParameters));
        index->SaveIndex(indexPathCpp);
    } catch (...) {
        for (auto & it : dataset) {
            delete it;
        }

        env->ReleaseIntArrayElements(idsJ, idsCpp, JNI_ABORT);
        HasExceptionInStack(env);
        throw;
    }
}

jlong knn_jni::nmslib_wrapper::LoadIndex(JNIEnv * env, jstring indexPathJ, jobject parametersJ) {

    if (indexPathJ == nullptr) {
        throw std::runtime_error("Index path cannot be null");
    }

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));

    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);

    // Get space type for this index
    jobject spaceTypeJ = knn_jni::GetJObjectFromMapOrThrow(parametersCpp, knn_jni::SPACE_TYPE);
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);

    // Parse query params
    std::vector<std::string> queryParams;

    if(parametersCpp.find("efSearch") != parametersCpp.end()) {
        auto efSearch = std::to_string(knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["efSearch"]));
        queryParams.push_back("efSearch=" + efSearch);
    }

    // Load index
    IndexWrapper * indexWrapper;
    try {
        indexWrapper = new IndexWrapper(spaceTypeCpp);
        indexWrapper->index->LoadIndex(indexPathCpp);
        indexWrapper->index->SetQueryTimeParams(similarity::AnyParams(queryParams));
    } catch (...) {
        delete indexWrapper;
        throw;
    }

    return (jlong) indexWrapper;
}

jobjectArray knn_jni::nmslib_wrapper::QueryIndex(JNIEnv * env, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ) {

    if (queryVectorJ == nullptr) {
        throw std::runtime_error("Query Vector cannot be null");
    }

    auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointerJ);

    if (indexWrapper == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    int dim	= knn_jni::GetJavaFloatArrayLength(env, queryVectorJ);

    float* rawQueryvector = env->GetFloatArrayElements(queryVectorJ, nullptr); // Have to call release on this
    if (rawQueryvector == nullptr) {
        knn_jni::HasExceptionInStack(env);
        throw std::runtime_error("Unable to get float elements from query vector");
    }

    std::unique_ptr<const similarity::Object> queryObject;
    try {
        queryObject.reset(new similarity::Object(-1, -1, dim*sizeof(float), rawQueryvector));
    } catch (...) {
        env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);
        knn_jni::HasExceptionInStack(env);
        throw;
    }
    env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);

    similarity::KNNQuery<float> knnQuery(*(indexWrapper->space), queryObject.get(), kJ);
    indexWrapper->index->Search(&knnQuery);

    std::unique_ptr<similarity::KNNQueue<float>> neighbors(knnQuery.Result()->Clone());
    int resultSize = neighbors->Size();

    jclass resultClass = knn_jni::FindClass(env,"org/opensearch/knn/index/KNNQueryResult");
    jmethodID allArgs = knn_jni::FindMethod(env, resultClass, "<init>", "(IF)V");

    jobjectArray results = env->NewObjectArray(resultSize, resultClass, nullptr);
    if (results == nullptr) {
        knn_jni::HasExceptionInStack(env);
        throw std::runtime_error("Unable to allocate results array");
    }

    jobject result;
    float distance;
    long id;
    for(int i = 0; i < resultSize; ++i) {
        distance = neighbors->TopDistance();
        id = neighbors->Pop()->id();
        result = env->NewObject(resultClass, allArgs, id, distance);
        knn_jni::HasExceptionInStack(env);
        if (result == nullptr) {
            throw std::runtime_error("Unable to create object");
        }
        env->SetObjectArrayElement(results, i, result);
        knn_jni::HasExceptionInStack(env);
    }
    return results;
}

void knn_jni::nmslib_wrapper::Free(jlong indexPointerJ) {
    auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointerJ);
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
        return std::string("negdotprod");
    }

    throw std::runtime_error("Invalid spaceType");
}
