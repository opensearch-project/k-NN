#include <cstring>
#include <limits>
#include <algorithm>
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

    JNI_UTIL.Initialize(env, vm);

    return KNN_SIMD_COMPUTE_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION);
    JNI_UTIL.Uninitialize(env);
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk
  (JNIEnv *env, jclass clazz, jintArray internalVectorIds, jfloatArray jscores, const jint numVectors) {
    if (numVectors <= 0) {
      return std::numeric_limits<float>::min();
    }

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      // Get pointers of vectorIds and scores
      jint* vectorIds = static_cast<jint*>(JNI_UTIL.GetPrimitiveArrayCritical(env, internalVectorIds, nullptr));
      knn_jni::JNIReleaseElements releaseVectorIds {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, internalVectorIds, vectorIds, 0);
      }};

      jfloat* scores = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jscores, nullptr));
      knn_jni::JNIReleaseElements releaseScores {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, jscores, scores, 0);
      }};

      // Bulk similarity calculation
      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      jfloat maxScore = *std::max_element(scores, scores + numVectors);
      return maxScore;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
      return 0.0f;   // value ignored if exception pending
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext
  (JNIEnv *env, jclass clazz, jfloatArray query, jlongArray addressAndSize, const jint nativeFunctionTypeOrd) {
    try {
      // Get raw pointer of query vector + size
      const jsize queryVecSize = JNI_UTIL.GetJavaFloatArrayLength(env, query);
      jfloat* queryVecPtr = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, query, nullptr));

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));

      // Save search context
      SimilarityFunction::saveSearchContext(
          (uint8_t*) queryVecPtr, sizeof(jfloat) * queryVecSize,
          queryVecSize,
          (int64_t*) mmapAddressAndSize, mmapAddressAndSizeLength,
          nativeFunctionTypeOrd);

      // Release query vector
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, query, queryVecPtr, 0);
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarity
  (JNIEnv *env, jclass clazz, const jint internalVectorId) {

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

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSQSearchContext
  (JNIEnv *env, jclass clazz, jbyteArray quantizedQuery,
   jfloat lowerInterval, jfloat upperInterval, jfloat additionalCorrection,
   jint quantizedComponentSum, jlongArray addressAndSize,
   jint functionTypeOrd, jint dimension, jfloat centroidDp) {
    try {
      // Get quantized query bytes
      const jsize queryByteSize = JNI_UTIL.GetJavaBytesArrayLength(env, quantizedQuery);
      jbyte* queryPtr = static_cast<jbyte*>(JNI_UTIL.GetPrimitiveArrayCritical(env, quantizedQuery, nullptr));
      knn_jni::JNIReleaseElements queryRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, quantizedQuery, queryPtr, 0);
      }};

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));
      knn_jni::JNIReleaseElements mmapAddressAndSizeRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
      }};

      // Store correction factors in tmpBuffer before calling saveSearchContext.
      // saveSearchContext will reset tmpBuffer at the beginning, so we need to call it first,
      // then write correction factors after.
      SimilarityFunction::saveSearchContext(
          reinterpret_cast<uint8_t*>(queryPtr), queryByteSize,
          dimension,
          reinterpret_cast<int64_t*>(mmapAddressAndSize), mmapAddressAndSizeLength,
          functionTypeOrd);

      // Now store correction factors in tmpBuffer (saveSearchContext clears it, then SQ_IP branch leaves it empty)
      SimdVectorSearchContext* ctx = SimilarityFunction::getSearchContext();
      ctx->tmpBuffer.resize(5 * sizeof(float));
      auto* correctionPtr = reinterpret_cast<float*>(ctx->tmpBuffer.data());
      correctionPtr[0] = lowerInterval;
      correctionPtr[1] = upperInterval;
      correctionPtr[2] = additionalCorrection;
      std::memcpy(&correctionPtr[3], &quantizedComponentSum, sizeof(int32_t));
      correctionPtr[4] = centroidDp;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}
