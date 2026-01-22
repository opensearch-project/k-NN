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

#include "org_opensearch_knn_jni_FaissService.h"

#include <jni.h>

#include <vector>

#include "faiss_wrapper.h"
#include "jni_util.h"
#include "faiss_stream_support.h"
#include "faiss/impl/FaissException.h"

static knn_jni::JNIUtil jniUtil;
static const jint KNN_FAISS_JNI_VERSION = JNI_VERSION_1_1;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jniUtil.Initialize(env);

    return KNN_FAISS_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_FAISS_JNI_VERSION);
    faiss::InterruptCallback::instance.get()->clear_instance();
    jniUtil.Uninitialize(env);
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initIndex(JNIEnv * env, jclass cls,
                                                                           jlong numDocs, jint dimJ,
                                                                           jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &indexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jlong numDocs, jint dimJ,
                                                                                 jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &binaryIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initByteIndex(JNIEnv * env, jclass cls,
                                                                               jlong numDocs, jint dimJ,
                                                                               jobject parametersJ)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        return knn_jni::faiss_wrapper::InitIndex(&jniUtil, env, numDocs, dimJ, parametersJ, &byteIndexService);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return (jlong)0;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                              jlong vectorsAddressJ, jint dimJ,
                                                                              jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &indexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToBinaryIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                    jlong vectorsAddressJ, jint dimJ,
                                                                                    jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &binaryIndexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_insertToByteIndex(JNIEnv * env, jclass cls, jintArray idsJ,
                                                                                  jlong vectorsAddressJ, jint dimJ,
                                                                                  jlong indexAddress, jint threadCount)
{
    try {
        std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
        knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
        knn_jni::faiss_wrapper::InsertToIndex(&jniUtil, env, idsJ, vectorsAddressJ, dimJ, indexAddress, threadCount, &byteIndexService);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        // NOTE: ADDING DELETE STATEMENT HERE CAUSES A CRASH!
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeIndex(JNIEnv * env,
                                                                           jclass cls,
                                                                           jlong indexAddress,
                                                                           jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::IndexService indexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &indexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeBinaryIndex(JNIEnv * env,
                                                                                 jclass cls,
                                                                                 jlong indexAddress,
                                                                                 jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::BinaryIndexService binaryIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &binaryIndexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_writeByteIndex(JNIEnv * env,
                                                                               jclass cls,
                                                                               jlong indexAddress,
                                                                               jobject output)
{
  try {
      std::unique_ptr<knn_jni::faiss_wrapper::FaissMethods> faissMethods(new knn_jni::faiss_wrapper::FaissMethods());
      knn_jni::faiss_wrapper::ByteIndexService byteIndexService(std::move(faissMethods));
      knn_jni::faiss_wrapper::WriteIndex(&jniUtil, env, output, indexAddress, &byteIndexService);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(JNIEnv * env,
                                                                                        jclass cls,
                                                                                        jintArray idsJ,
                                                                                        jlong vectorsAddressJ,
                                                                                        jint dimJ,
                                                                                        jobject output,
                                                                                        jbyteArray templateIndexJ,
                                                                                        jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateIndexFromTemplate(&jniUtil,
                                                        env,
                                                        idsJ,
                                                        vectorsAddressJ,
                                                        dimJ,
                                                        output,
                                                        templateIndexJ,
                                                        parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate(JNIEnv * env,
                                                                                              jclass cls,
                                                                                              jintArray idsJ,
                                                                                              jlong vectorsAddressJ,
                                                                                              jint dimJ,
                                                                                              jobject output,
                                                                                              jbyteArray templateIndexJ,
                                                                                              jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateBinaryIndexFromTemplate(&jniUtil,
                                                              env,
                                                              idsJ,
                                                              vectorsAddressJ,
                                                              dimJ,
                                                              output,
                                                              templateIndexJ,
                                                              parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createByteIndexFromTemplate(JNIEnv * env,
                                                                                            jclass cls,
                                                                                            jintArray idsJ,
                                                                                            jlong vectorsAddressJ,
                                                                                            jint dimJ,
                                                                                            jobject output,
                                                                                            jbyteArray templateIndexJ,
                                                                                            jobject parametersJ)
{
    try {
        knn_jni::faiss_wrapper::CreateByteIndexFromTemplate(&jniUtil,
                                                            env,
                                                            idsJ,
                                                            vectorsAddressJ,
                                                            dimJ,
                                                            output,
                                                            templateIndexJ,
                                                            parametersJ);
    }
    catch (const faiss::FaissException& e) {
        std::string errormsg = e.msg;
        std::size_t found = errormsg.find("computation interrupted");
        if (found != std::string::npos) {
            jniUtil.CatchAbortExceptionAndThrowJava(env);
        } else {
            jniUtil.CatchCppExceptionAndThrowJava(env);
        }
    }
    catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
  try {
      return knn_jni::faiss_wrapper::LoadIndex(&jniUtil, env, indexPathJ);
  } catch (...) {
      jniUtil.CatchCppExceptionAndThrowJava(env);
  }
  return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStream(JNIEnv * env,
                                                                                     jclass cls,
                                                                                     jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadIndexWithStream(
                 &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex(JNIEnv * env, jclass cls, jstring indexPathJ)
{
    try {
        return knn_jni::faiss_wrapper::LoadBinaryIndex(&jniUtil, env, indexPathJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndexWithStream(JNIEnv * env,
                                                                                           jclass cls,
                                                                                           jobject readStream)
{
    try {
        // Create a mediator locally.
        // Note that `indexInput` is `IndexInputWithBuffer` type.
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStream};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        // Pass IOReader to Faiss for loading vector index.
        return knn_jni::faiss_wrapper::LoadBinaryIndexWithStream(
            &faissOpenSearchIOReader);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }

    return NULL;
}
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndexWithStreamADCParams
(JNIEnv * env, jclass cls, jobject readStreamJ, jobject parametersJ) {
    try {
        knn_jni::stream::NativeEngineIndexInputMediator mediator {&jniUtil, env, readStreamJ};

        // Wrap the mediator with a glue code inheriting IOReader.
        knn_jni::stream::FaissOpenSearchIOReader faissOpenSearchIOReader {&mediator};

        return knn_jni::faiss_wrapper::LoadIndexWithStreamADCParams(&faissOpenSearchIOReader, &jniUtil, env, parametersJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired(JNIEnv * env,
                                                                                               jclass cls,
                                                                                               jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::IsSharedIndexStateRequired(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::InitSharedIndexState(indexPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
        (JNIEnv * env, jclass cls, jlong indexPointerJ, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::SetSharedIndexState(indexPointerJ, shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex(JNIEnv * env, jclass cls,
                                                                                   jlong indexPointerJ,
                                                                                   jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::QueryIndex(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, parentIdsJ);

    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter
  (JNIEnv * env, jclass cls, jlong indexPointerJ, jbyteArray queryVectorJ, jint kJ, jobject methodParamsJ, jlongArray filteredIdsJ, jint filterIdsTypeJ,  jintArray parentIdsJ) {

      try {
          return knn_jni::faiss_wrapper::QueryBinaryIndex_WithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, kJ, methodParamsJ, filteredIdsJ, filterIdsTypeJ, parentIdsJ);
      } catch (...) {
          jniUtil.CatchCppExceptionAndThrowJava(env);
      }
      return nullptr;

}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free(JNIEnv * env, jclass cls, jlong indexPointerJ, jboolean isBinaryIndexJ)
{
    try {
        return knn_jni::faiss_wrapper::Free(indexPointerJ, isBinaryIndexJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
        (JNIEnv * env, jclass cls, jlong shareIndexStatePointerJ)
{
    try {
        knn_jni::faiss_wrapper::FreeSharedIndexState(shareIndexStatePointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary(JNIEnv * env, jclass cls)
{
    try {
        knn_jni::faiss_wrapper::InitLibrary();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainBinaryIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainByteIndex(JNIEnv * env, jclass cls,
                                                                                 jobject parametersJ,
                                                                                 jint dimensionJ,
                                                                                 jlong trainVectorsPointerJ)
{
    try {
        return knn_jni::faiss_wrapper::TrainByteIndex(&jniUtil, env, parametersJ, dimensionJ, trainVectorsPointerJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors(JNIEnv * env, jclass cls,
                                                                                 jlong vectorsPointerJ,
                                                                                 jobjectArray vectorsJ)
{
    std::vector<float> *vect;
    if ((long) vectorsPointerJ == 0) {
        vect = new std::vector<float>;
    } else {
        vect = reinterpret_cast<std::vector<float>*>(vectorsPointerJ);
    }

    int dim = jniUtil.GetInnerDimensionOf2dJavaFloatArray(env, vectorsJ);
    auto dataset = jniUtil.Convert2dJavaObjectArrayToCppFloatVector(env, vectorsJ, dim);
    vect->insert(vect->begin(), dataset.begin(), dataset.end());

    return (jlong) vect;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex(JNIEnv * env, jclass cls,
                                                                                         jlong indexPointerJ,
                                                                                         jfloatArray queryVectorJ,
                                                                                         jfloat radiusJ, jobject methodParamsJ,
                                                                                         jint maxResultWindowJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearch(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter(JNIEnv * env, jclass cls,
                                                                                                   jlong indexPointerJ,
                                                                                                   jfloatArray queryVectorJ,
                                                                                                   jfloat radiusJ, jobject methodParamsJ, jint maxResultWindowJ,
                                                                                                   jlongArray filterIdsJ, jint filterIdsTypeJ, jintArray parentIdsJ)
{
    try {
        return knn_jni::faiss_wrapper::RangeSearchWithFilter(&jniUtil, env, indexPointerJ, queryVectorJ, radiusJ, methodParamsJ, maxResultWindowJ, filterIdsJ, filterIdsTypeJ, parentIdsJ);
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setMergeInterruptCallback(JNIEnv * env, jclass cls)
{
    try {
        faiss::InterruptCallback::instance.reset(
            new knn_jni::faiss_wrapper::OpenSearchMergeInterruptCallback(&jniUtil, env)
        );
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_testMergeInterruptCallback(JNIEnv * env, jclass cls)
{
    try {
        faiss::InterruptCallback::instance.get()->want_interrupt();
    } catch (...) {
        jniUtil.CatchCppExceptionAndThrowJava(env);
    }
}