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

#include "jni_util.h"

#include "methodfactory.h"
#include "space.h"
#include "spacefactory.h"

#include <jni.h>
#include <string>

namespace knn_jni {
    namespace nmslib_wrapper {
        // Create an index with ids and vectors. The configuration is defined by values in the Java map, parametersJ.
        // The index is serialized to indexPathJ.
        void CreateIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jintArray idsJ, jlong vectorsAddress, jint dim,
                         jstring indexPathJ, jobject parametersJ);

        // Load an index from indexPathJ into memory. Use parametersJ to set any query time parameters
        //
        // Return a pointer to the loaded index
        jlong LoadIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jstring indexPathJ, jobject parametersJ);

        // Execute a query against the index located in memory at indexPointerJ.
        //
        // Return an array of KNNQueryResults
        jobjectArray QueryIndex(knn_jni::JNIUtilInterface * jniUtil, JNIEnv * env, jlong indexPointerJ,
                                jfloatArray queryVectorJ, jint kJ, jobject methodParamsJ);

        // Free the index located in memory at indexPointerJ
        void Free(jlong indexPointer);

        // Perform required initialization operations for the library
        void InitLibrary();

        struct IndexWrapper {
            explicit IndexWrapper(const std::string& spaceType) {
                // Index gets constructed with a reference to data (see above) but is otherwise unused
                space.reset(similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(spaceType, similarity::AnyParams()));
                index.reset(similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "hnsw", spaceType, *space, data));
            }
            similarity::ObjectVector data;
            std::unique_ptr<similarity::Space<float>> space;
            std::unique_ptr<similarity::Index<float>> index;
        };
    }
}

#endif //OPENSEARCH_KNN_NMSLIB_WRAPPER_H
