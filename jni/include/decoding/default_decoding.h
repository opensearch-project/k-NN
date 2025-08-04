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
         * No SIMD support available on this platform.
         * @return false always.
         */
        jboolean isSIMDSupported();

        /**
         * Should never be called. Java fallback should be used instead.
         * Only declared to satisfy linkage.
         */
        jboolean convertFP16ToFP32(knn_jni::JNIUtilInterface *, JNIEnv* env, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset);
    }
}

#endif