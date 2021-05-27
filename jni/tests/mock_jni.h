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

#ifndef OPENSEARCH_KNN_MOCK_JNI_H
#define OPENSEARCH_KNN_MOCK_JNI_H

#include <jni.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace mock_jni {

    enum TypeMockParameterValue {
        TYPE_STRING,
        TYPE_INT,
        TYPE_MAP
    };

    struct MockParameterValue {
        int typeId;
        void * value;
    };

    struct MockParameter {
        const char * key;
        MockParameterValue value;
    };

    JNINativeInterface_ * GenerateMockJNINativeInterface();

    jsize MockGetArrayLength(jarray array);

    jboolean MockExceptionCheck();

    jint MockThrow(jthrowable jthrowable1);
}

#endif //OPENSEARCH_KNN_MOCK_JNI_H