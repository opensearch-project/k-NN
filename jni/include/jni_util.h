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

#ifndef OPENSEARCH_KNN_JNI_UTIL_H
#define OPENSEARCH_KNN_JNI_UTIL_H

#include <jni.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace knn_jni {
    // -------------------------- EXCEPTION HANDLING ----------------------------
    // Takes the name of a Java exception type and a message and throws the corresponding exception
    // to the JVM
    void ThrowJavaException(JNIEnv* env, const char* type = "", const char* message = "");

    // Checks if an exception occurred in the JVM and if so throws a C++ exception
    // This should be called after some calls to JNI functions
    void HasExceptionInStack(JNIEnv* env);

    // HasExceptionInStack with ability to specify message
    void HasExceptionInStack(JNIEnv* env, const std::string& message);

    // Catches a C++ exception and throws the corresponding exception to the JVM
    void CatchCppExceptionAndThrowJava(JNIEnv* env);

    // --------------------------------------------------------------------------

    // ------------------------------ JAVA FINDERS ------------------------------
    // Find a java class given a particular name
    jclass FindClass(JNIEnv * env, const std::string& className);

    // Find a java method given a particular class, name and signature
    jmethodID FindMethod(JNIEnv * env, jclass jClass, const std::string& methodName,
                         const std::string& methodSignature);

    // --------------------------------------------------------------------------

    // ------------------------- JAVA TO CPP CONVERTERS -------------------------
    // Returns cpp copied string from the Java string and releases the JNI Resource
    std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString);

    // Converts a java map to a cpp unordered_map<string, jobject>
    std::unordered_map<std::string, jobject> ConvertJavaMapToCppMap(JNIEnv *env, jobject parametersJ);

    // Convert a java object to cpp string, if applicable
    std::string ConvertJavaObjectToCppString(JNIEnv *env, jobject objectJ);

    // Convert a java object to a cpp integer, if applicable
    int ConvertJavaObjectToCppInteger(JNIEnv *env, jobject objectJ);

    std::vector<float> Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ, int dim);

    std::vector<int64_t> ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ);

    // --------------------------------------------------------------------------

    // ------------------------------ MISC HELPERS ------------------------------
    int GetInnerDimensionOf2dJavaArray(JNIEnv *env, jobjectArray array2dJ);

    int GetJavaObjectArrayLength(JNIEnv *env, jobjectArray arrayJ);

    int GetJavaIntArrayLength(JNIEnv *env, jintArray arrayJ);

    int GetJavaBytesArrayLength(JNIEnv *env, jbyteArray arrayJ);

    int GetJavaFloatArrayLength(JNIEnv *env, jfloatArray arrayJ);

    // --------------------------------------------------------------------------

    // ------------------------------- CONSTANTS --------------------------------
    extern const std::string FAISS_NAME;
    extern const std::string NMSLIB_NAME;

    extern const std::string ILLEGAL_ARGUMENT_PATH;

    extern const std::string SPACE_TYPE;
    extern const std::string METHOD;
    extern const std::string PARAMETERS;
    extern const std::string TRAINING_DATASET_SIZE_LIMIT;

    extern const std::string L2;
    extern const std::string L1;
    extern const std::string LINF;
    extern const std::string COSINESIMIL;
    extern const std::string INNER_PRODUCT;

    extern const std::string NPROBES;
    extern const std::string COARSE_QUANTIZER;
    extern const std::string EF_CONSTRUCTION;
    extern const std::string EF_SEARCH;

    // --------------------------------------------------------------------------
}

#endif //OPENSEARCH_KNN_JNI_UTIL_H
