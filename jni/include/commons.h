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

#ifndef OPENSEARCH_KNN_COMMONS_H
#define OPENSEARCH_KNN_COMMONS_H

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
         * append tells the method to keep appending to the existing vector. Passing the value as false will clear the vector
         * without reallocating new memory. This helps with reducing memory frangmentation and overhead of allocating
         * and deallocating when the memory address needs to be reused.
         *
         * CAUTION: The behavior is undefined if the memory address is deallocated and the method is called
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D float array containing data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @param append whether to append or start from index 0 when called subsequently with the same address
         * @return memory address of std::vector<float> where the data is stored.
         */
        jlong storeVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

        /**
         * This is utility function that can be used to store data in native memory. This function will allocate memory for
         * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
         * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
         * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
         * will throw Exception.
         *
         *  append tells the method to keep appending to the existing vector. Passing the value as false will clear the vector
         * without reallocating new memory. This helps with reducing memory frangmentation and overhead of allocating
         * and deallocating when the memory address needs to be reused.
         *
         * CAUTION: The behavior is undefined if the memory address is deallocated and the method is called
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D byte array containing binary data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @param append whether to append or start from index 0 when called subsequently with the same address
         * @return memory address of std::vector<uint8_t> where the data is stored.
         */
        jlong storeBinaryVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

        /**
        * This is utility function that can be used to store signed int8 data in native memory. This function will allocate memory for
        * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
        * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
        * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
        * will throw Exception.
        *
        * @param memoryAddress The address of the memory location where data will be stored.
        * @param data 2D byte array containing int8 data to be stored in native memory.
        * @param initialCapacity The initial capacity of the memory location.
        * @param append whether to append or start from index 0 when called subsequently with the same address
        * @return memory address of std::vector<int8_t> where the data is stored.
        */
        jlong storeByteVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

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
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeBinaryVectorData(long, byte[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeBinaryVectorData(jlong);

        /**
         * Extracts query time efSearch from method parameters
         **/
        int getIntegerMethodParameter(JNIEnv *, knn_jni::JNIUtilInterface *, std::unordered_map<std::string, jobject>, std::string, int);

        /**
         * Converts an array of FP32 values to FP16 values.
         * @param fp32Array The input array of FP32 values.
         * @param fp16Array The output array of FP16 values.
         * @param count The number of elements in the arrays.
         */
        void convertFP32ToFP16(
            knn_jni::JNIUtilInterface *, JNIEnv *, jfloatArray fp32Array, jbyteArray fp16Array, jint count
        );

        /**
         * Converts an array of FP16 values to FP32 values.
         * @param fp16Array The input array of FP16 values.
         * @param fp32Array The output array of FP32 values.
         * @param count The number of elements in the arrays.
         * @param offset The byte offset for each element in the arrays.
         */
        void convertFP16ToFP32(
            knn_jni::JNIUtilInterface *, JNIEnv *, jbyteArray fp16Array, jfloatArray fp32Array, jint count, jint offset
        );
    }
}

#endif