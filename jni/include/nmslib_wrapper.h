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

#ifndef OPENSEARCH_KNN_NMSLIB_WRAPPER_H
#define OPENSEARCH_KNN_NMSLIB_WRAPPER_H

#include <jni.h>

namespace knn_jni {
    namespace nmslib_wrapper {
        void createIndex(JNIEnv *, jintArray, jobjectArray, jstring, jobject);

        jlong loadIndex(JNIEnv *, jstring, jobject);

        jobjectArray queryIndex(JNIEnv *, jlong, jfloatArray, jint, jobject);

        void free(jlong);

        void initLibrary();
    }
}

#endif //OPENSEARCH_KNN_NMSLIB_WRAPPER_H
