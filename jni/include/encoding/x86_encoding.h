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

#ifndef OPENSEARCH_KNN_X86_ENCODING_H
#define OPENSEARCH_KNN_X86_ENCODING_H

#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace encoding {
        /**
         * Checks if the x86 architecture supports SIMD operations.
         * @return true if SIMD is supported, false otherwise.
         */
        jboolean isSIMDSupported();

        /**
         * Converts an array of FP32 values to FP16 values.
         * @param fp32Array The input array of FP32 values.
         * @param fp16Array The output array of FP16 values.
         * @param count The number of elements in the arrays.
         * @return JNI_TRUE on success, JNI_FALSE on failure.
         */
        jboolean convertFP32ToFP16(JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count);
    }
}

#endif