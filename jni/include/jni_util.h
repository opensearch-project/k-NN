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
#include <cstdint>

namespace knn_jni {

    // Interface for making calls to JNI
    struct JNIUtilInterface {
        // -------------------------- EXCEPTION HANDLING ----------------------------
        // Takes the name of a Java exception type and a message and throws the corresponding exception
        // to the JVM
        virtual void ThrowJavaException(JNIEnv* env, const char* type, const char* message) = 0;

        // Checks if an exception occurred in the JVM and if so throws a C++ exception
        // This should be called after some calls to JNI functions
        virtual void HasExceptionInStack(JNIEnv* env) = 0;

        // HasExceptionInStack with ability to specify message
        virtual void HasExceptionInStack(JNIEnv* env, const std::string& message) = 0;

        // Catches a C++ exception and throws the corresponding exception to the JVM
        virtual void CatchCppExceptionAndThrowJava(JNIEnv* env) = 0;
        // --------------------------------------------------------------------------

        // ------------------------------ JAVA FINDERS ------------------------------
        // Find a java class given a particular name
        virtual jclass FindClass(JNIEnv * env, const std::string& className) = 0;

        // Find a java method given a particular class, name and signature
        virtual jmethodID FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) = 0;

        // --------------------------------------------------------------------------

        // ------------------------- JAVA TO CPP CONVERTERS -------------------------
        // Returns cpp copied string from the Java string and releases the JNI Resource
        virtual std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) = 0;

        // Converts a java map to a cpp unordered_map<string, jobject>
        //TODO: My concern with this function is that it will make a lot of calls between the JVM. A few options
        // to explore are:
        // 1. Passing a json string and parsing it in CPP layer
        // 2. Caching some of the method and class calls
        virtual std::unordered_map<std::string, jobject> ConvertJavaMapToCppMap(JNIEnv *env, jobject parametersJ) = 0;

        // Convert a java object to cpp string, if applicable
        virtual std::string ConvertJavaObjectToCppString(JNIEnv *env, jobject objectJ) = 0;

        // Convert a java object to a cpp integer, if applicable
        virtual int ConvertJavaObjectToCppInteger(JNIEnv *env, jobject objectJ) = 0;

        virtual std::vector<float> Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ,
                                                                            int dim) = 0;

        virtual void Convert2dJavaObjectArrayAndStoreToFloatVector(JNIEnv *env, jobjectArray array2dJ,
                                                                   int dim, std::vector<float> *vect ) = 0;
        virtual void Convert2dJavaObjectArrayAndStoreToBinaryVector(JNIEnv *env, jobjectArray array2dJ,
                                                                   int dim, std::vector<uint8_t> *vect ) = 0;
        virtual void Convert2dJavaObjectArrayAndStoreToByteVector(JNIEnv *env, jobjectArray array2dJ,
                                                                           int dim, std::vector<int8_t> *vect ) = 0;

        virtual std::vector<int64_t> ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ) = 0;

        // --------------------------------------------------------------------------

        // ------------------------------ MISC HELPERS ------------------------------
        virtual int GetInnerDimensionOf2dJavaFloatArray(JNIEnv *env, jobjectArray array2dJ) = 0;

        virtual int GetInnerDimensionOf2dJavaByteArray(JNIEnv *env, jobjectArray array2dJ) = 0;

        virtual int GetJavaObjectArrayLength(JNIEnv *env, jobjectArray arrayJ) = 0;

        virtual int GetJavaIntArrayLength(JNIEnv *env, jintArray arrayJ) = 0;

        virtual int GetJavaLongArrayLength(JNIEnv *env, jlongArray arrayJ) = 0;

        virtual int GetJavaBytesArrayLength(JNIEnv *env, jbyteArray arrayJ) = 0;

        virtual int GetJavaFloatArrayLength(JNIEnv *env, jfloatArray arrayJ) = 0;

        // ---------------------------- Direct calls to JNIEnv ----------------------------

        virtual void DeleteLocalRef(JNIEnv *env, jobject obj) = 0;

        virtual jbyte * GetByteArrayElements(JNIEnv *env, jbyteArray array, jboolean * isCopy) = 0;

        virtual jfloat * GetFloatArrayElements(JNIEnv *env, jfloatArray array, jboolean * isCopy) = 0;

        virtual jint * GetIntArrayElements(JNIEnv *env, jintArray array, jboolean * isCopy) = 0;

        virtual jlong * GetLongArrayElements(JNIEnv *env, jlongArray array, jboolean * isCopy) = 0;

        virtual jobject GetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index) = 0;

        virtual jobject NewObject(JNIEnv *env, jclass clazz, jmethodID methodId, int id, float distance) = 0;

        virtual jobjectArray NewObjectArray(JNIEnv *env, jsize len, jclass clazz, jobject init) = 0;

        virtual jbyteArray NewByteArray(JNIEnv *env, jsize len) = 0;

        virtual void ReleaseByteArrayElements(JNIEnv *env, jbyteArray array, jbyte *elems, int mode) = 0;

        virtual void ReleaseFloatArrayElements(JNIEnv *env, jfloatArray array, jfloat *elems, int mode) = 0;

        virtual void ReleaseIntArrayElements(JNIEnv *env, jintArray array, jint *elems, jint mode) = 0;

        virtual void ReleaseLongArrayElements(JNIEnv *env, jlongArray array, jlong *elems, jint mode) = 0;

        virtual void SetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index, jobject val) = 0;

        virtual void SetByteArrayRegion(JNIEnv *env, jbyteArray array, jsize start, jsize len, const jbyte * buf) = 0;

        virtual jobject GetObjectField(JNIEnv * env, jobject obj, jfieldID fieldID) = 0;

        virtual jclass FindClassFromJNIEnv(JNIEnv * env, const char *name) = 0;

        virtual jmethodID GetMethodID(JNIEnv * env, jclass clazz, const char *name, const char *sig) = 0;

        virtual jfieldID GetFieldID(JNIEnv * env, jclass clazz, const char *name, const char *sig) = 0;

        virtual void * GetPrimitiveArrayCritical(JNIEnv * env, jarray array, jboolean *isCopy) = 0;

        virtual void ReleasePrimitiveArrayCritical(JNIEnv * env, jarray array, void *carray, jint mode) = 0;

        virtual jint CallIntMethodInt(JNIEnv * env, jobject obj, jmethodID methodID, int intArg) = 0;

        // --------------------------------------------------------------------------
    };

    jobject GetJObjectFromMapOrThrow(std::unordered_map<std::string, jobject> map, std::string key);

    // Class that implements JNIUtilInterface methods
    class JNIUtil final : public JNIUtilInterface {
    public:
        // Initialize and Uninitialize methods are used for caching/cleaning up Java classes and methods
        void Initialize(JNIEnv* env);
        void Uninitialize(JNIEnv* env);

        void ThrowJavaException(JNIEnv* env, const char* type = "", const char* message = "") final;
        void HasExceptionInStack(JNIEnv* env) final;
        void HasExceptionInStack(JNIEnv* env, const std::string& message) final;
        void CatchCppExceptionAndThrowJava(JNIEnv* env) final;
        jclass FindClass(JNIEnv * env, const std::string& className) final;
        jmethodID FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) final;
        std::string ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) final;
        std::unordered_map<std::string, jobject> ConvertJavaMapToCppMap(JNIEnv *env, jobject parametersJ) final;
        std::string ConvertJavaObjectToCppString(JNIEnv *env, jobject objectJ) final;
        int ConvertJavaObjectToCppInteger(JNIEnv *env, jobject objectJ) final;
        std::vector<float> Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ, int dim) final;
        std::vector<int64_t> ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ) final;
        int GetInnerDimensionOf2dJavaFloatArray(JNIEnv *env, jobjectArray array2dJ) final;
        int GetInnerDimensionOf2dJavaByteArray(JNIEnv *env, jobjectArray array2dJ) final;
        int GetJavaObjectArrayLength(JNIEnv *env, jobjectArray arrayJ) final;
        int GetJavaIntArrayLength(JNIEnv *env, jintArray arrayJ) final;
        int GetJavaLongArrayLength(JNIEnv *env, jlongArray arrayJ) final;
        int GetJavaBytesArrayLength(JNIEnv *env, jbyteArray arrayJ) final;
        int GetJavaFloatArrayLength(JNIEnv *env, jfloatArray arrayJ) final;

        void DeleteLocalRef(JNIEnv *env, jobject obj) final;
        jbyte * GetByteArrayElements(JNIEnv *env, jbyteArray array, jboolean * isCopy) final;
        jfloat * GetFloatArrayElements(JNIEnv *env, jfloatArray array, jboolean * isCopy) final;
        jint * GetIntArrayElements(JNIEnv *env, jintArray array, jboolean * isCopy) final;
        jlong * GetLongArrayElements(JNIEnv *env, jlongArray array, jboolean * isCopy) final;
        jobject GetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index) final;
        jobject NewObject(JNIEnv *env, jclass clazz, jmethodID methodId, int id, float distance) final;
        jobjectArray NewObjectArray(JNIEnv *env, jsize len, jclass clazz, jobject init) final;
        jbyteArray NewByteArray(JNIEnv *env, jsize len) final;
        void ReleaseByteArrayElements(JNIEnv *env, jbyteArray array, jbyte *elems, int mode) final;
        void ReleaseFloatArrayElements(JNIEnv *env, jfloatArray array, jfloat *elems, int mode) final;
        void ReleaseIntArrayElements(JNIEnv *env, jintArray array, jint *elems, jint mode) final;
        void ReleaseLongArrayElements(JNIEnv *env, jlongArray array, jlong *elems, jint mode) final;
        void SetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index, jobject val) final;
        void SetByteArrayRegion(JNIEnv *env, jbyteArray array, jsize start, jsize len, const jbyte * buf) final;
        void Convert2dJavaObjectArrayAndStoreToFloatVector(JNIEnv *env, jobjectArray array2dJ, int dim, std::vector<float> *vect) final;
        void Convert2dJavaObjectArrayAndStoreToBinaryVector(JNIEnv *env, jobjectArray array2dJ, int dim, std::vector<uint8_t> *vect) final;
        void Convert2dJavaObjectArrayAndStoreToByteVector(JNIEnv *env, jobjectArray array2dJ, int dim, std::vector<int8_t> *vect) final;
        jobject GetObjectField(JNIEnv * env, jobject obj, jfieldID fieldID) final;
        jclass FindClassFromJNIEnv(JNIEnv * env, const char *name) final;
        jmethodID GetMethodID(JNIEnv * env, jclass clazz, const char *name, const char *sig) final;
        jfieldID GetFieldID(JNIEnv * env, jclass clazz, const char *name, const char *sig) final;
        jint CallIntMethodInt(JNIEnv * env, jobject obj, jmethodID methodID, int intArg) final;
        void * GetPrimitiveArrayCritical(JNIEnv * env, jarray array, jboolean *isCopy) final;
        void ReleasePrimitiveArrayCritical(JNIEnv * env, jarray array, void *carray, jint mode) final;

    private:
        std::unordered_map<std::string, jclass> cachedClasses;
        std::unordered_map<std::string, jmethodID> cachedMethods;
    };

    // ------------------------------- CONSTANTS --------------------------------
    extern const std::string FAISS_NAME;
    extern const std::string NMSLIB_NAME;

    extern const std::string ILLEGAL_ARGUMENT_PATH;

    extern const std::string SPACE_TYPE;
    extern const std::string METHOD;
    extern const std::string INDEX_DESCRIPTION;
    extern const std::string PARAMETERS;
    extern const std::string TRAINING_DATASET_SIZE_LIMIT;
    extern const std::string INDEX_THREAD_QUANTITY;

    extern const std::string L2;
    extern const std::string L1;
    extern const std::string LINF;
    extern const std::string COSINESIMIL;
    extern const std::string INNER_PRODUCT;
    extern const std::string NEG_DOT_PRODUCT;
    extern const std::string HAMMING;

    extern const std::string NPROBES;
    extern const std::string COARSE_QUANTIZER;
    extern const std::string M;
    extern const std::string M_NMSLIB;
    extern const std::string EF_CONSTRUCTION;
    extern const std::string EF_CONSTRUCTION_NMSLIB;
    extern const std::string EF_SEARCH;

    // --------------------------------------------------------------------------
}

#endif //OPENSEARCH_KNN_JNI_UTIL_H
