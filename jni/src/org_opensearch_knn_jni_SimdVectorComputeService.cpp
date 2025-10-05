#include "org_opensearch_knn_jni_SimdVectorComputeService.h"
#include "jni_util.h"
#include "simd/similarity_function/similarity_function.h"

static knn_jni::JNIUtil JNI_UTIL;
static constexpr jint KNN_SIMD_COMPUTE_JNI_VERSION = JNI_VERSION_1_1;

using knn_jni::simd::similarity_function::SimilarityFunction;
using knn_jni::simd::similarity_function::SimdVectorSearchContext;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    JNI_UTIL.Initialize(env);

    return KNN_SIMD_COMPUTE_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION);
    JNI_UTIL.Uninitialize(env);
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_bulkDistanceCalculation
  (JNIEnv *env, jclass clazz, jintArray internalVectorIds, jfloatArray jscores, jint numVectors) {

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      // Get pointers
      jint* vectorIds = static_cast<jint*>(env->GetPrimitiveArrayCritical(internalVectorIds, nullptr));
      jfloat* scores = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jscores, nullptr));

      // Bulk similarity calculation
      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      // Release pinned pointers
      env->ReleasePrimitiveArrayCritical(internalVectorIds, vectorIds, 0);
      env->ReleasePrimitiveArrayCritical(jscores, scores, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext
  (JNIEnv *env, jclass clazz, jfloatArray query, jlongArray addressAndSize, jint nativeFunctionTypeOrd) {
    try {
      // Get raw pointer of query vector + size
      jsize queryVecSize = env->GetArrayLength(query);
      jfloat* queryVecPtr = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(query, nullptr));

      // Get mmap address and size
      jsize mmapAddressAndSizeLength = env->GetArrayLength(addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(env->GetPrimitiveArrayCritical(addressAndSize, nullptr));

      // Save search context
      SimilarityFunction::saveSearchContext(
          (uint8_t*) queryVecPtr, sizeof(jfloat) * queryVecSize,
          queryVecSize,
          (int64_t*) mmapAddressAndSize, mmapAddressAndSizeLength,
          nativeFunctionTypeOrd);

      // Release query vector
      env->ReleasePrimitiveArrayCritical(query, queryVecPtr, 0);
      env->ReleasePrimitiveArrayCritical(addressAndSize, mmapAddressAndSize, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSingleVector
  (JNIEnv *env, jclass clazz, jint internalVectorId) {

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();

      // Single vector similarity calculation.
      return srchContext->similarityFunction->calculateSimilarity(srchContext, internalVectorId);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }

    return 0;
}
