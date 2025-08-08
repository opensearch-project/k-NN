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
#include "encoding/encoding.h"

jboolean knn_jni::encoding::isSIMDSupported() {
    return JNI_FALSE;
}

jboolean knn_jni::encoding::convertFP32ToFP16(knn_jni::JNIUtilInterface *jniUtil, JNIEnv*, jfloatArray, jbyteArray, jint) {
    return JNI_FALSE;
}
