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

#ifndef OPENSEARCH_KNN_ARM_DECODING_H
#define OPENSEARCH_KNN_ARM_DECODING_H

#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace decoding {
        /**
         * Checks if the ARM architecture supports SIMD operations.
         * @return true if SIMD is supported, false otherwise.
         */
        jboolean isSIMDSupported();

        /**
         * Converts an array of FP16 values to FP32 values.
         * @param fp16Array The input array of FP16 values.
         * @param fp32Array The output array of FP32 values.
         * @param count The number of elements to convert.
         * @param offset Offset in the FP16 input array in bytes.
         * @return JNI_TRUE on success, JNI_FALSE on failure.
         */
        jboolean convertFP16ToFP32(knn_jni::JNIUtilInterface *, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset);
    }
}

#endif