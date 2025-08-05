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
#include <iostream>

#include <cstdint>

#include "jni_util.h"
#include "decoding/decoding.h"

jboolean knn_jni::decoding::isSIMDSupported() {
    return JNI_FALSE;
}

jboolean knn_jni::decoding::convertFP16ToFP32(knn_jni::JNIUtilInterface *jniUtil, JNIEnv*, jbyteArray, jfloatArray, jint, jint) {
    std::cerr << "[KNN] Warning: convertFP16ToFP32 called on unsupported platform. Java fallback expected.\n";
    return JNI_FALSE;
}