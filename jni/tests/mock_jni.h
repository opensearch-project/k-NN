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

// TODO: Add utilities to directly create and load indices from library w/o our library
namespace mock_jni {

    JNINativeInterface_ * GenerateMockJNINativeInterface();

}

#endif //OPENSEARCH_KNN_MOCK_JNI_H