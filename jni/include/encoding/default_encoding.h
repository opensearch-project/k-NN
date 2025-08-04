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

#ifndef OPENSEARCH_KNN_DEFAULT_ENCODING_H
#define OPENSEARCH_KNN_DEFAULT_ENCODING_H

#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace encoding {
        /**
         * No SIMD support available on this platform.
         * @return false always.
         */
        jboolean isSIMDSupported();

        /**
         * Should never be called. Java fallback should be used instead.
         * Only declared to satisfy linkage.
         */
        jboolean convertFP32ToFP16(knn_jni::JNIUtilInterface *, JNIEnv* env, jfloatArray fp32Array, jbyteArray fp16Array, jint count);
    }
}

#endif