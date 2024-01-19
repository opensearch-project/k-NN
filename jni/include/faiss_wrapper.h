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

#ifndef OPENSEARCH_KNN_FAISS_WRAPPER_H
#define OPENSEARCH_KNN_FAISS_WRAPPER_H

#include "jni_util.h"
#include <jni.h>

namespace knn_jni {
    namespace faiss_wrapper {
        // Create an index with ids and vectors. The configuration is defined by values in the Java map, parametersJ.
        // The index is serialized to indexPathJ.
        void CreateIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ, jobjectArray vectorsJ,
                         jstring indexPathJ, jobject parametersJ);

        // Create an index with ids and vectors. Instead of creating a new index, this function creates the index
        // based off of the template index passed in. The index is serialized to indexPathJ.
        void CreateIndexFromTemplate(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ,
                                     jobjectArray vectorsJ, jstring indexPathJ, jbyteArray templateIndexJ,
                                     jobject parametersJ);

        // Load an index from indexPathJ into memory.
        //
        // Return a pointer to the loaded index
        jlong LoadIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ);

        // Execute a query against the index located in memory at indexPointerJ.
        //
        // Return an array of KNNQueryResults
        jobjectArray QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                jfloatArray queryVectorJ, jint kJ, jintArray parentIdsJ);

        // Execute a query against the index located in memory at indexPointerJ along with Filters
        //
        // Return an array of KNNQueryResults
        jobjectArray QueryIndex_WithFilter(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                                                jfloatArray queryVectorJ, jint kJ, jintArray filterIdsJ, jintArray parentIdsJ);

        // Free the index located in memory at indexPointerJ
        void Free(jlong indexPointer);

        // Perform initilization operations for the library
        void InitLibrary();

        // Create an empty index defined by the values in the Java map, parametersJ. Train the index with
        // the vector of floats located at trainVectorsPointerJ.
        //
        // Return the serialized representation
        jbyteArray TrainIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jobject parametersJ, jint dimension,
                              jlong trainVectorsPointerJ);
    }
}

#endif //OPENSEARCH_KNN_FAISS_WRAPPER_H
