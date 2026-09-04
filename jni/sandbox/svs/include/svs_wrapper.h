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

#ifndef OPENSEARCH_KNN_SVS_WRAPPER_H
#define OPENSEARCH_KNN_SVS_WRAPPER_H

#include "jni_util.h"

#include "faiss/impl/io.h"
#include "faiss/impl/AuxIndexStructures.h"

#include <jni.h>
#include <iostream>

namespace knn_jni {
    namespace svs_wrapper {
        // Creates the index from the parameters map and returns its build context.
        jlong InitIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong numDocs, jint dimJ, jobject parametersJ);

        // Buffers a batch of vectors into the build context.
        void InsertToIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ, jlong vectorsAddressJ,
                           jint dimJ, jlong indexAddressJ, jint threadCount);

        // Builds the graph, serializes the index, and frees the build context.
        void WriteIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject output, jlong indexAddressJ);

        jlong LoadIndexWithStream(faiss::IOReader* ioReader);

        jobjectArray QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ);

        jobjectArray QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                           jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ,
                                           jlongArray filterIdsJ, jint filterIdsTypeJ);

        // Radial search within radiusJ (faiss-domain, > 0), capped at maxResultWindowJ.
        jobjectArray RangeSearch(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                 jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ,
                                 jint maxResultWindowJ);

        jobjectArray RangeSearch_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                            jfloatArray queryVectorJ, jfloat radiusJ, jobject methodParamsJ,
                                            jint maxResultWindowJ, jlongArray filterIdsJ, jint filterIdsTypeJ);

        void Free(jlong indexPointerJ);

        void InitLibrary();

        bool IsLvqLeanvecEnabled();

        faiss::MetricType TranslateSpaceToMetric(const std::string& spaceType);

        // Interrupt callback that aborts a native build when the driving Lucene merge is aborted.
        struct OpenSearchMergeInterruptCallback : faiss::InterruptCallback {

            OpenSearchMergeInterruptCallback(JNIUtil *jniUtil) {
                jutil = jniUtil;
                JNIEnv* jenv = jutil->GetJNICurrentEnv();
                mergeHelperClass = jniUtil->FindClass(jenv, "org/apache/lucene/index/MergeAbortChecker");
                isAbortedMethod = jniUtil->FindMethod(jenv, "org/apache/lucene/index/MergeAbortChecker", "isMergeAborted");
            }

            bool want_interrupt() override {
                JNIEnv* jenv = jutil->GetJNICurrentEnv();
                if (jenv == nullptr) {
                    std::cerr << "JNIEnv not found\n";
                    return false;
                }
                if (mergeHelperClass == nullptr) {
                    std::cerr << "MergeAbortChecker class not found\n";
                    return false;
                }
                if (isAbortedMethod == nullptr) {
                    std::cerr << "isMergeAborted method not found\n";
                    return false;
                }
                return (bool) jenv->CallStaticBooleanMethod(mergeHelperClass, isAbortedMethod);
            }

            JNIUtil *jutil;
            jclass mergeHelperClass;
            jmethodID isAbortedMethod;
        };
    }
}

#endif //OPENSEARCH_KNN_SVS_WRAPPER_H
