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
#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace commons {
        /**
         * This is utility function that can be used to store data in native memory. This function will allocate memory for
         * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
         * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
         * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
         * will throw Exception.
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D float array containing data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @return memory address of std::vector<float> where the data is stored.
         */
        jlong storeVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong);

        /**
         * This is utility function that can be used to store data in native memory. This function will allocate memory for
         * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
         * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
         * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
         * will throw Exception.
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D byte array containing data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @return memory address of std::vector<uint8_t> where the data is stored.
         */
        jlong storeByteVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong);

        /**
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeVectorData(long, float[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeVectorData(jlong);

        /**
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeByteVectorData(long, byte[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeByteVectorData(jlong);

        /**
         * Extracts query time efSearch from method parameters
         **/
        int getIntegerMethodParameter(JNIEnv *, knn_jni::JNIUtilInterface *, std::unordered_map<std::string, jobject>, std::string, int);
    }
}
