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

#include "jni_util.h"
#include "simd/fp16_codec/fp16_codec.h"

namespace knn_jni::simd::fp16_codec {

jboolean isSIMDSupported() {
    return JNI_FALSE;
}

jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jfloatArray fp32Array, jbyteArray fp16Array, jint count) {
    return JNI_FALSE;
}

jboolean decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                           jbyteArray fp16Array, jint offset, jfloatArray fp32Array, jint count) {
    return JNI_FALSE;
}

}  // namespace knn_jni::simd::fp16_codec
