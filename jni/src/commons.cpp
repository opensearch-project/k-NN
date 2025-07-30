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
#include <jni.h>

#include <cstdint>
#include <vector>

#include "jni_util.h"
#include "commons.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

jlong knn_jni::commons::storeVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<float> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<float>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToFloatVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeBinaryVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<uint8_t> *vect;
    if ((long) memoryAddressJ == 0) {
        vect = new std::vector<uint8_t>();
        vect->reserve((long)initialCapacityJ);
    } else {
        vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
        vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToBinaryVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

jlong knn_jni::commons::storeByteVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ, jboolean appendJ) {
    std::vector<int8_t> *vect;
    if (memoryAddressJ == 0) {
        vect = new std::vector<int8_t>();
        vect->reserve(static_cast<long>(initialCapacityJ));
    } else {
        vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
    }

    if (appendJ == JNI_FALSE) {
            vect->clear();
    }

    int dim = jniUtil->GetInnerDimensionOf2dJavaByteArray(env, dataJ);
    jniUtil->Convert2dJavaObjectArrayAndStoreToByteVector(env, dataJ, dim, vect);

    return (jlong) vect;
}

void knn_jni::commons::freeVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeBinaryVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<uint8_t>*>(memoryAddressJ);
        delete vect;
    }
}

void knn_jni::commons::freeByteVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<int8_t>*>(memoryAddressJ);
        delete vect;
    }
}

int knn_jni::commons::getIntegerMethodParameter(JNIEnv * env, knn_jni::JNIUtilInterface * jniUtil, std::unordered_map<std::string, jobject> methodParams, std::string methodParam, int defaultValue) {
    if (methodParams.empty()) {
        return defaultValue;
    }
    auto efSearchIt = methodParams.find(methodParam);
    if (efSearchIt != methodParams.end()) {
        return jniUtil->ConvertJavaObjectToCppInteger(env, methodParams[methodParam]);
    }

    return defaultValue;
}

// FP32 → FP16
void knn_jni::commons::convertFP32ToFP16(knn_jni::JNIUtilInterface* jniUtil,
                                         JNIEnv* env,
                                         jfloatArray fp32Array,
                                         jbyteArray fp16Array,
                                         jint count) {
    if (count <= 0) return;

    jfloat* src_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* dst_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));
    const float* src = reinterpret_cast<const float*>(src_f32);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bytes);

    int i = 0;

#if defined(__aarch64__)
    // ARM NEON bulk 8-wide
    for (; i + 8 <= count; i += 8) {
        float32x4_t v0 = vld1q_f32(&src[i + 0]);
        float32x4_t v1 = vld1q_f32(&src[i + 4]);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 0]), h0);
        vst1_f16(reinterpret_cast<__fp16*>(&dst[i + 4]), h1);
    }
    // tail via NEON scalar broadcast
    for (; i < count; ++i) {
        float32x4_t sv = vdupq_n_f32(src[i]);
        float16x4_t hv = vcvt_f16_f32(sv);
        __fp16 lane = vget_lane_f16(hv, 0);
        dst[i] = *reinterpret_cast<const uint16_t*>(&lane);
    }

#elif defined(__x86_64__)
  #if defined(__AVX512F__)
    // x86 AVX-512 bulk 16-wide
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(&src[i]);
        __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), h);
    }
  #elif defined(__AVX2__) && defined(__F16C__)
    // x86 AVX2+F16C bulk 8-wide
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(&src[i]);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[i]), h);
    }
  #else
    #error "x86_64 must support AVX512F or AVX2+F16C"
  #endif
    // tail via F16C scalar
    for (; i < count; ++i) {
        __m128 sv = _mm_set_ss(src[i]);
        __m128i hv = _mm_cvtps_ph(sv, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(hv));
    }
#else
    #error "Only aarch64 or x86_64 supported"
#endif

    jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, dst_bytes, 0);
    jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, src_f32, JNI_ABORT);
}


// FP16 → FP32
void knn_jni::commons::convertFP16ToFP32(knn_jni::JNIUtilInterface* jniUtil,
                                         JNIEnv* env,
                                         jbyteArray fp16Array,
                                         jfloatArray fp32Array,
                                         jint count,
                                         jint offset) {
    if (count <= 0) return;

    jfloat* dst_f32 = reinterpret_cast<jfloat*>(jniUtil->GetPrimitiveArrayCritical(env, fp32Array, nullptr));
    jbyte* src_bytes = reinterpret_cast<jbyte*>(jniUtil->GetPrimitiveArrayCritical(env, fp16Array, nullptr));
    float* dst = reinterpret_cast<float*>(dst_f32);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes + offset);

    int i = 0;

#if defined(__aarch64__)
    // ARM NEON bulk 8-wide
    for (; i + 8 <= count; i += 8) {
        float16x4_t h0 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 0]));
        float16x4_t h1 = vld1_f16(reinterpret_cast<const __fp16*>(&src[i + 4]));
        float32x4_t v0 = vcvt_f32_f16(h0);
        float32x4_t v1 = vcvt_f32_f16(h1);
        vst1q_f32(&dst[i + 0], v0);
        vst1q_f32(&dst[i + 4], v1);
    }
    // tail via NEON scalar broadcast
    for (; i < count; ++i) {
        __fp16 half = *reinterpret_cast<const __fp16*>(&src[i]);
        float32x4_t fv = vcvt_f32_f16(vdup_n_f16(half));
        dst[i] = vgetq_lane_f32(fv, 0);
    }

#elif defined(__x86_64__)
  #if defined(__AVX512F__)
    // x86 AVX-512 bulk 16-wide
    for (; i + 16 <= count; i += 16) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i]));
        __m512 v = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(&dst[i], v);
    }
  #elif defined(__AVX2__) && defined(__F16C__)
    // x86 AVX2+F16C bulk 8-wide
    for (; i + 8 <= count; i += 8) {
        if (i + 64 < count) {
            _mm_prefetch(reinterpret_cast<const char*>(&src[i + 64]), _MM_HINT_T0);
        }
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[i]));
        __m256 v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(&dst[i], v);
    }
  #else
    #error "x86_64 must support AVX512F or AVX2+F16C"
  #endif
    // tail via F16C scalar
    for (; i < count; ++i) {
        __m128i h = _mm_cvtsi32_si128(src[i]);
        __m128 v = _mm_cvtph_ps(h);
        dst[i] = _mm_cvtss_f32(v);
    }
#else
    #error "Only aarch64 or x86_64 supported"
#endif

    jniUtil->ReleasePrimitiveArrayCritical(env, fp32Array, dst_f32, 0);
    jniUtil->ReleasePrimitiveArrayCritical(env, fp16Array, src_bytes, JNI_ABORT);
}