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
#include <new>
#include <stdexcept>
#include <string>
#include <vector>


void knn_jni::ThrowJavaException(JNIEnv* env, const char* type, const char* message) {
    jclass newExcCls = env->FindClass(type);
    if (newExcCls != nullptr) {
        env->ThrowNew(newExcCls, message);
    }
    // If newExcCls isn't found, NoClassDefFoundError will be thrown
}

void knn_jni::HasExceptionInStack(JNIEnv* env)
{
    knn_jni::HasExceptionInStack(env, "Exception in jni occurred");
}

void knn_jni::HasExceptionInStack(JNIEnv* env, const std::string& message)
{
    if (env->ExceptionCheck() == JNI_TRUE) {
        throw std::runtime_error(message);
    }
}

void knn_jni::CatchCppExceptionAndThrowJava(JNIEnv* env)
{
    try {
        throw;
    }
    catch (const std::bad_alloc& rhs) {
        ThrowJavaException(env, "java/io/IOException", rhs.what());
    }
    catch (const std::runtime_error& re) {
        ThrowJavaException(env, "java/lang/Exception", re.what());
    }
    catch (const std::exception& e) {
        ThrowJavaException(env, "java/lang/Exception", e.what());
    }
    catch (...) {
        ThrowJavaException(env, "java/lang/Exception", "Unknown exception occurred");
    }
}

jclass knn_jni::FindClass(JNIEnv * env, const std::string& className) {
    jclass jClass = env->FindClass(className.c_str());
    knn_jni::HasExceptionInStack(env, "Error looking up \"" + className + "\"");
    if (jClass == nullptr) {
        throw std::runtime_error("Unable to load class \"" + className + "\"");
    }
    return jClass;
}

jmethodID knn_jni::FindMethod(JNIEnv * env, jclass jClass, const std::string& methodName, const std::string& methodSignature) {
    jmethodID methodId = env->GetMethodID(jClass, methodName.c_str(), methodSignature.c_str());
    knn_jni::HasExceptionInStack(env, "Error looking up \"" + methodName + "\" method");
    if (jClass == nullptr) {
        throw std::runtime_error("Unable to find \"" + methodName + "\" method");
    }
    return methodId;
}

//TODO: My concern with this code is that it is making a lot of calls back and forth between the JVM. A few options
// to explore are:
// 1. Passing a json string and parsing it in CPP layer
// 2. Caching some of the method and class calls
std::unordered_map<std::string, jobject> knn_jni::ConvertJavaMapToCppMap(JNIEnv *env, jobject parametersJ) {
    // Here, we parse parametersJ, which is a java Map<String, Object>. In order to implement this, I referred to
    // https://stackoverflow.com/questions/4844022/jni-create-hashmap

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    // Load all of the class and methods to iterate over a map
    jclass mapClassJ = knn_jni::FindClass(env, "java/util/Map");

    jmethodID entrySetMethodJ = knn_jni::FindMethod(env, mapClassJ, "entrySet", "()Ljava/util/Set;");

    jobject parametersEntrySetJ = env->CallObjectMethod(parametersJ, entrySetMethodJ);
    knn_jni::HasExceptionInStack(env, R"(Unable to call "entrySet" method on "java/util/Map")");

    jclass setClassJ = knn_jni::FindClass(env, "java/util/Set");

    jmethodID iteratorJ = knn_jni::FindMethod(env, setClassJ, "iterator", "()Ljava/util/Iterator;");

    jclass iteratorClassJ = knn_jni::FindClass(env, "java/util/Iterator");

    jobject iterJ = env->CallObjectMethod(parametersEntrySetJ, iteratorJ);
    knn_jni::HasExceptionInStack(env, R"(Call to "iterator" method failed")");

    jmethodID hasNextMethodJ = knn_jni::FindMethod(env, iteratorClassJ, "hasNext", "()Z");
    jmethodID nextMethodJ = knn_jni::FindMethod(env, iteratorClassJ, "next", "()Ljava/lang/Object;");

    jclass entryClassJ = knn_jni::FindClass(env, "java/util/Map$Entry");

    jmethodID getKeyMethodJ = knn_jni::FindMethod(env, entryClassJ, "getKey", "()Ljava/lang/Object;");

    jmethodID getValueMethodJ = knn_jni::FindMethod(env, entryClassJ, "getValue", "()Ljava/lang/Object;");

    // Iterate over the java map and add entries to cpp unordered map
    jobject entryJ;
    jstring keyJ;
    std::string keyCpp;
    jobject valueJ;
    std::unordered_map<std::string, jobject> parametersCpp;
    while (env->CallBooleanMethod(iterJ, hasNextMethodJ)) {
        entryJ = env->CallObjectMethod(iterJ, nextMethodJ);
        knn_jni::HasExceptionInStack(env, R"(Could not call "next" method")");

        keyJ = (jstring) env->CallObjectMethod(entryJ, getKeyMethodJ);
        knn_jni::HasExceptionInStack(env, R"(Could not call "getKey" method")");

        keyCpp = knn_jni::ConvertJavaStringToCppString(env, keyJ);

        valueJ = env->CallObjectMethod(entryJ, getValueMethodJ);
        knn_jni::HasExceptionInStack(env, R"(Could not call "getValue" method")");

        parametersCpp[keyCpp] = valueJ;
    }

    knn_jni::HasExceptionInStack(env, R"(Could not call "hasNext" method")");

    return parametersCpp;
}

std::string knn_jni::ConvertJavaObjectToCppString(JNIEnv *env, jobject objectJ) {
    return knn_jni::ConvertJavaStringToCppString(env, (jstring) objectJ);
}

std::string knn_jni::ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) {
    if (javaString == nullptr) {
        throw std::runtime_error("String cannot be null");
    }

    const char *cString = env->GetStringUTFChars(javaString, nullptr);
    if (cString == nullptr) {
        HasExceptionInStack(env);
    }
    std::string cppString(cString);
    env->ReleaseStringUTFChars(javaString, cString);
    return cppString;
}

int knn_jni::ConvertJavaObjectToCppInteger(JNIEnv *env, jobject objectJ) {

    if (objectJ == nullptr) {
        throw std::runtime_error("Object cannot be null");
    }

    jclass integerClass = knn_jni::FindClass(env, "java/lang/Integer");
    jmethodID intValue = knn_jni::FindMethod(env, integerClass, "intValue", "()I");

    if (!env->IsInstanceOf(objectJ, integerClass)) {
        throw std::runtime_error("Cannot call IntMethod on non-integer class");
    }

    int intCpp = env->CallIntMethod(objectJ, intValue);
    knn_jni::HasExceptionInStack(env, "Could not call \"intValue\" method on Integer");
    return intCpp;
}

std::vector<float> knn_jni::Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ, int dim) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    std::vector<float> floatVectorCpp;
    int numVectors = env->GetArrayLength(array2dJ);
    knn_jni::HasExceptionInStack(env);

    for (int i = 0; i < numVectors; ++i) {
        auto vectorArray = (jfloatArray)env->GetObjectArrayElement(array2dJ, i);
        knn_jni::HasExceptionInStack(env);

        if (dim != env->GetArrayLength(vectorArray)) {
            throw std::runtime_error("Dimension of vectors is inconsistent");
        }

        float* vector = env->GetFloatArrayElements(vectorArray, nullptr);
        knn_jni::HasExceptionInStack(env);
        for(int j = 0; j < dim; ++j) {
            floatVectorCpp.push_back(vector[j]);
        }
        env->ReleaseFloatArrayElements(vectorArray, vector, 0);
    }
    knn_jni::HasExceptionInStack(env);
    env->DeleteLocalRef(array2dJ);
    return floatVectorCpp;
}

int knn_jni::GetInnerDimensionOf2dJavaArray(JNIEnv *env, jobjectArray array2dJ) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    if (env->GetArrayLength(array2dJ) <= 0) {
        return 0;
    }

    auto vectorArray = (jfloatArray)env->GetObjectArrayElement(array2dJ, 0);
    knn_jni::HasExceptionInStack(env);
    int dim = env->GetArrayLength(vectorArray);
    knn_jni::HasExceptionInStack(env);
    return dim;
}

int knn_jni::GetJavaObjectArrayLength(JNIEnv *env, jobjectArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    knn_jni::HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::GetJavaIntArrayLength(JNIEnv *env, jintArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    knn_jni::HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::GetJavaBytesArrayLength(JNIEnv *env, jbyteArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    knn_jni::HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::GetJavaFloatArrayLength(JNIEnv *env, jfloatArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    knn_jni::HasExceptionInStack(env, "Unable to get array length");
    return length;
}

jobject knn_jni::GetJObjectFromMapOrThrow(std::unordered_map<std::string, jobject> map, std::string key) {
    if(map.find(key) == map.end()) {
        throw std::runtime_error(key + " not found");
    }
    return map[key];
}


std::vector<int64_t> knn_jni::ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    std::vector<int64_t> vectorCpp;
    int* arrayCpp = env->GetIntArrayElements(arrayJ, nullptr);
    vectorCpp.reserve(env->GetArrayLength(arrayJ));
    for(int i = 0; i < env->GetArrayLength(arrayJ); ++i) {
        vectorCpp.push_back(arrayCpp[i]);
    }
    env->ReleaseIntArrayElements(arrayJ, arrayCpp, 0);
    return vectorCpp;
}

//TODO: This potentially should use const char *
const std::string knn_jni::FAISS_NAME = "faiss";
const std::string knn_jni::NMSLIB_NAME = "nmslib";
const std::string knn_jni::ILLEGAL_ARGUMENT_PATH = "java/lang/IllegalArgumentException";

const std::string knn_jni::SPACE_TYPE = "spaceType";
const std::string knn_jni::METHOD = "method";
const std::string knn_jni::PARAMETERS = "parameters";
const std::string knn_jni::TRAINING_DATASET_SIZE_LIMIT = "training_dataset_size_limit";

const std::string knn_jni::L2 = "l2";
const std::string knn_jni::L1 = "l1";
const std::string knn_jni::LINF = "linf";
const std::string knn_jni::COSINESIMIL = "cosinesimil";
const std::string knn_jni::INNER_PRODUCT = "innerproduct";

const std::string knn_jni::NPROBES = "nprobes";
const std::string knn_jni::COARSE_QUANTIZER = "coarse_quantizer";
const std::string knn_jni::EF_CONSTRUCTION = "ef_construction";
const std::string knn_jni::EF_SEARCH = "ef_search";
