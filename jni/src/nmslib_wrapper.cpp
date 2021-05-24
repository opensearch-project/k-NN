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

void knn_jni::nmslib_wrapper::createIndex(JNIEnv * env, jintArray idsJ, jobjectArray vectorsJ, jstring indexPathJ,
                                          jobject parametersJ) {

    try {
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
    } catch (...) {
        CatchCppExceptionAndThrowJava(env);
    }

    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);
    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));
    // Get space type for this index
    if(parametersCpp.find(knn_jni::SPACE_TYPE) == parametersCpp.end()) {
        throw std::runtime_error("Space type not found");
    }
    jobject spaceTypeJ = parametersCpp[knn_jni::SPACE_TYPE];
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);
    similarity::Space<float>* space = similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceTypeCpp,similarity::AnyParams());

    // Read in data set
    int* idsCpp = nullptr;
    similarity::ObjectVector dataset;
    try {
        int numVectors = knn_jni::GetJavaObjectArrayLength(env, vectorsJ);
        int numIds = knn_jni::GetJavaIntArrayLength(env, idsJ);

        if (numIds != numVectors) {
            throw std::runtime_error("Number of IDs does not match number of vectors");
        }

        idsCpp = env->GetIntArrayElements(idsJ, nullptr);

        int dim = knn_jni::GetInnerDimensionOf2dJavaArray(env, vectorsJ);
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

        similarity::Index<float>* index = similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceTypeCpp, *space, dataset);


        std::vector<std::string> indexParameters;

        if(parametersCpp.find("ef_construction") != parametersCpp.end()) {
            auto efConstruction = knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["ef_construction"]);
            indexParameters.push_back("efConstruction=" + std::to_string(efConstruction));
        }

        if(parametersCpp.find("m") != parametersCpp.end()) {
            auto m = knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["m"]);
            indexParameters.push_back("M=" + std::to_string(m));
        }

        index->CreateIndex(similarity::AnyParams(indexParameters));
        index->SaveIndex(indexPathCpp);

        for (auto & it : dataset) {
            delete it;
        }
        delete index;
        delete space;

    } catch (...) {
        for (auto & it : dataset) {
            delete it;
        }

        env->ReleaseIntArrayElements(idsJ, idsCpp, JNI_ABORT);
        HasExceptionInStack(env);
        throw;
    }
}

jlong knn_jni::nmslib_wrapper::loadIndex(JNIEnv * env, jstring indexPathJ, jobject parametersJ) {
    std::string indexPathCpp(ConvertJavaStringToCppString(env, indexPathJ));

    auto parametersCpp = knn_jni::ConvertJavaMapToCppMap(env, parametersJ);

    // Get space type for this index
    if(parametersCpp.find(knn_jni::SPACE_TYPE) == parametersCpp.end()) {
        throw std::runtime_error("Space type not found");
    }
    jobject spaceTypeJ = parametersCpp[knn_jni::SPACE_TYPE];
    std::string spaceTypeCpp(knn_jni::ConvertJavaObjectToCppString(env, spaceTypeJ));
    spaceTypeCpp = TranslateSpaceType(spaceTypeCpp);
    auto *indexWrapper = new IndexWrapper(spaceTypeCpp);
    indexWrapper->index->LoadIndex(indexPathCpp);

    // Parse and set query params
    std::vector<std::string> queryParams;

    //TODO: efSearch should be integer
    if(parametersCpp.find("efSearch") != parametersCpp.end()) {
        auto efSearch = std::to_string(knn_jni::ConvertJavaObjectToCppInteger(env, parametersCpp["efSearch"]));
        queryParams.push_back("efSearch=" + efSearch);
    }
    indexWrapper->index->SetQueryTimeParams(similarity::AnyParams(queryParams));

    return (jlong) indexWrapper;
}

jobjectArray knn_jni::nmslib_wrapper::queryIndex(JNIEnv * env, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ,
                                                 jobject parametersJ) {
    auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointerJ);

    if (indexWrapper == nullptr) {
        throw std::runtime_error("Invalid pointer to index");
    }

    int dim	= knn_jni::GetJavaFloatArrayLength(env, queryVectorJ);
    float* rawQueryvector;
    std::unique_ptr<const similarity::Object> queryObject;

    try {
        rawQueryvector = env->GetFloatArrayElements(queryVectorJ, nullptr); // Have to call release on this
        knn_jni::HasExceptionInStack(env);
        queryObject.reset(new similarity::Object(-1, -1, dim*sizeof(float), rawQueryvector));
        env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);
        knn_jni::HasExceptionInStack(env);
    } catch (...) {
        env->ReleaseFloatArrayElements(queryVectorJ, rawQueryvector, JNI_ABORT);
        knn_jni::HasExceptionInStack(env);
        throw;
    }

    similarity::KNNQuery<float> knnQuery(*(indexWrapper->space), queryObject.get(), kJ);
    indexWrapper->index->Search(&knnQuery);

    std::unique_ptr<similarity::KNNQueue<float>> neighbors(knnQuery.Result()->Clone());
    int resultSize = neighbors->Size();

    jclass resultClass = knn_jni::FindClass(env,"org/opensearch/knn/index/KNNQueryResult");
    jmethodID allArgs = knn_jni::FindMethod(env, resultClass, "<init>", "(IF)V");

    jobjectArray results = env->NewObjectArray(resultSize, resultClass, nullptr);
    knn_jni::HasExceptionInStack(env);
    if (results == nullptr) {
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

void knn_jni::nmslib_wrapper::free(jlong indexPointerJ) {
    auto *indexWrapper = reinterpret_cast<IndexWrapper*>(indexPointerJ);
    delete indexWrapper;
}

void knn_jni::nmslib_wrapper::initLibrary() {
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
