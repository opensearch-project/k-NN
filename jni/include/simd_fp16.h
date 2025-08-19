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

#ifndef OPENSEARCH_KNN_CODEC_FP16_H
#define OPENSEARCH_KNN_CODEC_FP16_H

#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace simd {
        /**
         * Checks if the system architecture supports SIMD operations.
         * @return true if SIMD is supported, false otherwise.
         */
        jboolean isSIMDSupported();
    }

    namespace codec {
        namespace fp16 {
            /**
             * Converts an array of FP32 values to FP16 values.
             * @param jniUtil JNI utility interface.
             * @param env JNI environment pointer.
             * @param fp32Array The input array of FP32 values.
             * @param fp16Array The output array of FP16 values.
             * @param count The number of elements in the arrays.
             * @return JNI_TRUE on success, JNI_FALSE on failure.
             */
            jboolean encodeFp32ToFp16(knn_jni::JNIUtilInterface *, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count);

            /**
             * Converts an array of FP16 values to FP32 values.
             * @param jniUtil JNI utility interface.
             * @param env JNI environment pointer.
             * @param fp16Array The input array of FP16 values.
             * @param fp32Array The output array of FP32 values.
             * @param count The number of elements to convert.
             * @param offset Offset in the FP16 input array in bytes.
             * @return JNI_TRUE on success, JNI_FALSE on failure.
             */
            jboolean decodeFp16ToFp32(knn_jni::JNIUtilInterface *, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset);
        }
    }
}

#endif