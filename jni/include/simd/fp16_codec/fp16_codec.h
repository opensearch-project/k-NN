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

#ifndef OPENSEARCH_KNN_SIMD_FP16_CODEC_H
#define OPENSEARCH_KNN_SIMD_FP16_CODEC_H

#include <jni.h>
#include "jni_util.h"

namespace knn_jni::simd::fp16_codec {

    /**
     * Checks if the system architecture supports SIMD operations.
     * @return JNI_TRUE if SIMD is supported, JNI_FALSE otherwise.
     */
    jboolean isSIMDSupported();

    /**
     * Converts an array of FP32 values to FP16 values.
     * @param jniUtil JNI utility interface.
     * @param env JNI environment pointer.
     * @param fp32Array The input array of FP32 values.
     * @param fp16Array The output array of FP16 values.
     * @param count The number of elements to convert.
     * @return JNI_TRUE on success, JNI_FALSE on failure.
     */
    jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                               jfloatArray fp32Array, jbyteArray fp16Array, jint count);

    /**
     * Converts an array of FP16 values (packed as bytes) to FP32 values.
     * @param jniUtil JNI utility interface.
     * @param env JNI environment pointer.
     * @param fp16Array The input array of FP16 values, packed 2 bytes per element.
     * @param offset Byte offset into fp16Array where the encoded values start.
     * @param fp32Array The output array of FP32 values.
     * @param count The number of elements to convert.
     * @return JNI_TRUE on success, JNI_FALSE on failure.
     */
    jboolean decodeFp16ToFp32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv* env,
                               jbyteArray fp16Array, jint offset, jfloatArray fp32Array, jint count);

}

#endif  // OPENSEARCH_KNN_SIMD_FP16_CODEC_H
