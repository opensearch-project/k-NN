/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H
#define OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H

#include <string>

namespace knn_jni {
    // Intel SVS Vamana parameter names. These live ONLY in the isolated SVS JNI library
    // (libopensearchknn_svs); the main faiss JNI library carries no SVS symbols. They were
    // intentionally removed from jni_util when SVS was pulled out of the shared layer.
    extern const std::string CONSTRUCTION_WINDOW_SIZE;
    extern const std::string ALPHA;
    extern const std::string SEARCH_WINDOW_SIZE;
    extern const std::string SEARCH_BUFFER_CAPACITY;
}

#endif //OPENSEARCH_KNN_JNI_SVS_CONSTANTS_H
