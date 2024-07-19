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


void knn_jni::JNIUtil::Initialize(JNIEnv *env) {
    // Followed recommendation from this SO post: https://stackoverflow.com/a/13940735
    jclass tempLocalClassRef;

    tempLocalClassRef = env->FindClass("java/io/IOException");
    this->cachedClasses["java/io/IOException"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/lang/Exception");
    this->cachedClasses["java/lang/Exception"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/util/Map");
    this->cachedClasses["java/util/Map"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["java/util/Map:entrySet"] = env->GetMethodID(tempLocalClassRef, "entrySet", "()Ljava/util/Set;");
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/util/Set");
    this->cachedClasses["java/util/Set"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["java/util/Set:iterator"] = env->GetMethodID(tempLocalClassRef, "iterator", "()Ljava/util/Iterator;");
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/util/Iterator");
    this->cachedClasses["java/util/Iterator"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["java/util/Iterator:hasNext"] = env->GetMethodID(tempLocalClassRef, "hasNext", "()Z");
    this->cachedMethods["java/util/Iterator:next"] = env->GetMethodID(tempLocalClassRef, "next", "()Ljava/lang/Object;");
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/lang/Object");
    this->cachedClasses["java/lang/Object"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/util/Map$Entry");
    this->cachedClasses["java/util/Map$Entry"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["java/util/Map$Entry:getKey"] = env->GetMethodID(tempLocalClassRef, "getKey", "()Ljava/lang/Object;");
    this->cachedMethods["java/util/Map$Entry:getValue"] = env->GetMethodID(tempLocalClassRef, "getValue", "()Ljava/lang/Object;");
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("java/lang/Integer");
    this->cachedClasses["java/lang/Integer"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["java/lang/Integer:intValue"] = env->GetMethodID(tempLocalClassRef, "intValue", "()I");
    env->DeleteLocalRef(tempLocalClassRef);

    tempLocalClassRef = env->FindClass("org/opensearch/knn/index/query/KNNQueryResult");
    this->cachedClasses["org/opensearch/knn/index/query/KNNQueryResult"] = (jclass) env->NewGlobalRef(tempLocalClassRef);
    this->cachedMethods["org/opensearch/knn/index/query/KNNQueryResult:<init>"] = env->GetMethodID(tempLocalClassRef, "<init>", "(IF)V");
    env->DeleteLocalRef(tempLocalClassRef);
}

void knn_jni::JNIUtil::Uninitialize(JNIEnv* env) {
    // Delete all classes that are now global refs
    for (auto & cachedClasse : this->cachedClasses) {
        env->DeleteGlobalRef(cachedClasse.second);
    }
    this->cachedClasses.clear();
    this->cachedMethods.clear();
}

void knn_jni::JNIUtil::ThrowJavaException(JNIEnv* env, const char* type, const char* message) {
    jclass newExcCls = env->FindClass(type);
    if (newExcCls != nullptr) {
        env->ThrowNew(newExcCls, message);
    }
    // If newExcCls isn't found, NoClassDefFoundError will be thrown
}

void knn_jni::JNIUtil::HasExceptionInStack(JNIEnv* env) {
    this->HasExceptionInStack(env, "Exception in jni occurred");
}

void knn_jni::JNIUtil::HasExceptionInStack(JNIEnv* env, const std::string& message) {
    if (env->ExceptionCheck() == JNI_TRUE) {
        throw std::runtime_error(message);
    }
}

void knn_jni::JNIUtil::CatchCppExceptionAndThrowJava(JNIEnv* env)
{
    try {
        throw;
    }
    catch (const std::bad_alloc& rhs) {
        this->ThrowJavaException(env, "java/io/IOException", rhs.what());
    }
    catch (const std::runtime_error& re) {
        this->ThrowJavaException(env, "java/lang/Exception", re.what());
    }
    catch (const std::exception& e) {
        this->ThrowJavaException(env, "java/lang/Exception", e.what());
    }
    catch (...) {
        this->ThrowJavaException(env, "java/lang/Exception", "Unknown exception occurred");
    }
}

jclass knn_jni::JNIUtil::FindClass(JNIEnv * env, const std::string& className) {
    if (this->cachedClasses.find(className) == this->cachedClasses.end()) {
        throw std::runtime_error("Unable to load class \"" + className + "\"");
    }

    return this->cachedClasses[className];
}

jmethodID knn_jni::JNIUtil::FindMethod(JNIEnv * env, const std::string& className, const std::string& methodName) {
    std::string key = className + ":" + methodName;
    if (this->cachedMethods.find(key) == this->cachedMethods.end()) {
        throw std::runtime_error("Unable to find \"" + methodName + "\" method");
    }

    return this->cachedMethods[key];
}

std::unordered_map<std::string, jobject> knn_jni::JNIUtil::ConvertJavaMapToCppMap(JNIEnv *env, jobject parametersJ) {
    // Here, we parse parametersJ, which is a java Map<String, Object>. In order to implement this, I referred to
    // https://stackoverflow.com/questions/4844022/jni-create-hashmap. All java references are local, so they will be
    // freed when the native method returns

    if (parametersJ == nullptr) {
        throw std::runtime_error("Parameters cannot be null");
    }

    // Load all of the methods to iterate over a map
    jmethodID entrySetMethodJ = this->FindMethod(env, "java/util/Map", "entrySet");

    jobject parametersEntrySetJ = env->CallObjectMethod(parametersJ, entrySetMethodJ);
    this->HasExceptionInStack(env, R"(Unable to call "entrySet" method on "java/util/Map")");
    jmethodID iteratorJ = this->FindMethod(env, "java/util/Set", "iterator");
    jobject iterJ = env->CallObjectMethod(parametersEntrySetJ, iteratorJ);
    this->HasExceptionInStack(env, R"(Call to "iterator" method failed")");

    jmethodID hasNextMethodJ = this->FindMethod(env, "java/util/Iterator", "hasNext");
    jmethodID nextMethodJ = this->FindMethod(env, "java/util/Iterator", "next");
    jmethodID getKeyMethodJ = this->FindMethod(env, "java/util/Map$Entry", "getKey");
    jmethodID getValueMethodJ = this->FindMethod(env, "java/util/Map$Entry", "getValue");

    // Iterate over the java map and add entries to cpp unordered map
    jobject entryJ;
    jstring keyJ;
    std::string keyCpp;
    jobject valueJ;
    std::unordered_map<std::string, jobject> parametersCpp;
    while (env->CallBooleanMethod(iterJ, hasNextMethodJ)) {
        entryJ = env->CallObjectMethod(iterJ, nextMethodJ);
        this->HasExceptionInStack(env, R"(Could not call "next" method")");

        keyJ = (jstring) env->CallObjectMethod(entryJ, getKeyMethodJ);
        this->HasExceptionInStack(env, R"(Could not call "getKey" method")");

        keyCpp = this->ConvertJavaStringToCppString(env, keyJ);

        valueJ = env->CallObjectMethod(entryJ, getValueMethodJ);
        this->HasExceptionInStack(env, R"(Could not call "getValue" method")");

        parametersCpp[keyCpp] = valueJ;

        env->DeleteLocalRef(entryJ);
        env->DeleteLocalRef(keyJ);
    }

    this->HasExceptionInStack(env, R"(Could not call "hasNext" method")");

    return parametersCpp;
}

std::string knn_jni::JNIUtil::ConvertJavaObjectToCppString(JNIEnv *env, jobject objectJ) {
    return this->ConvertJavaStringToCppString(env, (jstring) objectJ);
}

std::string knn_jni::JNIUtil::ConvertJavaStringToCppString(JNIEnv * env, jstring javaString) {
    if (javaString == nullptr) {
        throw std::runtime_error("String cannot be null");
    }

    const char *cString = env->GetStringUTFChars(javaString, nullptr);
    if (cString == nullptr) {
        this->HasExceptionInStack(env, "Unable to convert java string to cpp string");

        // Will only reach here if there is no exception in the stack, but the call failed
        throw std::runtime_error("Unable to convert java string to cpp string");
    }
    std::string cppString(cString);
    env->ReleaseStringUTFChars(javaString, cString);
    return cppString;
}

int knn_jni::JNIUtil::ConvertJavaObjectToCppInteger(JNIEnv *env, jobject objectJ) {

    if (objectJ == nullptr) {
        throw std::runtime_error("Object cannot be null");
    }

    jclass integerClass = this->FindClass(env, "java/lang/Integer");
    jmethodID intValue = this->FindMethod(env, "java/lang/Integer", "intValue");

    if (!env->IsInstanceOf(objectJ, integerClass)) {
        throw std::runtime_error("Cannot call IntMethod on non-integer class");
    }

    int intCpp = env->CallIntMethod(objectJ, intValue);
    this->HasExceptionInStack(env, "Could not call \"intValue\" method on Integer");
    return intCpp;
}

std::vector<float> knn_jni::JNIUtil::Convert2dJavaObjectArrayToCppFloatVector(JNIEnv *env, jobjectArray array2dJ,
                                                                              int dim) {
    std::vector<float> vect;
    Convert2dJavaObjectArrayAndStoreToFloatVector(env, array2dJ, dim, &vect);
    return vect;
}

void knn_jni::JNIUtil::Convert2dJavaObjectArrayAndStoreToFloatVector(JNIEnv *env, jobjectArray array2dJ,
                                                                     int dim, std::vector<float> *vect) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int numVectors = env->GetArrayLength(array2dJ);
    this->HasExceptionInStack(env);

    for (int i = 0; i < numVectors; ++i) {
        auto vectorArray = (jfloatArray)env->GetObjectArrayElement(array2dJ, i);
        this->HasExceptionInStack(env, "Unable to get object array element");

        if (dim != env->GetArrayLength(vectorArray)) {
            throw std::runtime_error("Dimension of vectors is inconsistent");
        }

        float* vector = env->GetFloatArrayElements(vectorArray, nullptr);
        if (vector == nullptr) {
            this->HasExceptionInStack(env);
            throw std::runtime_error("Unable to get float array elements");
        }

        for(int j = 0; j < dim; ++j) {
            vect->push_back(vector[j]);
        }
        env->ReleaseFloatArrayElements(vectorArray, vector, JNI_ABORT);
    }
    this->HasExceptionInStack(env);
    env->DeleteLocalRef(array2dJ);
}

void knn_jni::JNIUtil::Convert2dJavaObjectArrayAndStoreToByteVector(JNIEnv *env, jobjectArray array2dJ,
                                                                     int dim, std::vector<uint8_t> *vect) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int numVectors = env->GetArrayLength(array2dJ);
    this->HasExceptionInStack(env);

    for (int i = 0; i < numVectors; ++i) {
        auto vectorArray = (jbyteArray)env->GetObjectArrayElement(array2dJ, i);
        this->HasExceptionInStack(env, "Unable to get object array element");

        if (dim != env->GetArrayLength(vectorArray)) {
            throw std::runtime_error("Dimension of vectors is inconsistent");
        }

        uint8_t* vector = reinterpret_cast<uint8_t*>(env->GetByteArrayElements(vectorArray, nullptr));
        if (vector == nullptr) {
            this->HasExceptionInStack(env);
            throw std::runtime_error("Unable to get byte array elements");
        }

        for(int j = 0; j < dim; ++j) {
            vect->push_back(vector[j]);
        }
        env->ReleaseByteArrayElements(vectorArray, reinterpret_cast<int8_t*>(vector), JNI_ABORT);
    }
    this->HasExceptionInStack(env);
    env->DeleteLocalRef(array2dJ);
}

std::vector<int64_t> knn_jni::JNIUtil::ConvertJavaIntArrayToCppIntVector(JNIEnv *env, jintArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int numElements = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");

    int* arrayCpp = env->GetIntArrayElements(arrayJ, nullptr);
    if (arrayCpp == nullptr) {
        this->HasExceptionInStack(env, "Unable to get integer array elements");
        throw std::runtime_error("Unable to get integer array elements");
    }

    std::vector<int64_t> vectorCpp;
    vectorCpp.reserve(numElements);
    for(int i = 0; i < numElements; ++i) {
        vectorCpp.push_back(arrayCpp[i]);
    }
    env->ReleaseIntArrayElements(arrayJ, arrayCpp, JNI_ABORT);
    return vectorCpp;
}

int knn_jni::JNIUtil::GetInnerDimensionOf2dJavaFloatArray(JNIEnv *env, jobjectArray array2dJ) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    if (env->GetArrayLength(array2dJ) <= 0) {
        return 0;
    }

    auto vectorArray = (jfloatArray)env->GetObjectArrayElement(array2dJ, 0);
    this->HasExceptionInStack(env);
    int dim = env->GetArrayLength(vectorArray);
    this->HasExceptionInStack(env);
    return dim;
}

int knn_jni::JNIUtil::GetInnerDimensionOf2dJavaByteArray(JNIEnv *env, jobjectArray array2dJ) {

    if (array2dJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    if (env->GetArrayLength(array2dJ) <= 0) {
        return 0;
    }

    auto vectorArray = (jbyteArray)env->GetObjectArrayElement(array2dJ, 0);
    this->HasExceptionInStack(env);
    int dim = env->GetArrayLength(vectorArray);
    this->HasExceptionInStack(env);
    return dim;
}

int knn_jni::JNIUtil::GetJavaObjectArrayLength(JNIEnv *env, jobjectArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::JNIUtil::GetJavaIntArrayLength(JNIEnv *env, jintArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::JNIUtil::GetJavaLongArrayLength(JNIEnv *env, jlongArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::JNIUtil::GetJavaBytesArrayLength(JNIEnv *env, jbyteArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");
    return length;
}

int knn_jni::JNIUtil::GetJavaFloatArrayLength(JNIEnv *env, jfloatArray arrayJ) {

    if (arrayJ == nullptr) {
        throw std::runtime_error("Array cannot be null");
    }

    int length = env->GetArrayLength(arrayJ);
    this->HasExceptionInStack(env, "Unable to get array length");
    return length;
}

void knn_jni::JNIUtil::DeleteLocalRef(JNIEnv *env, jobject obj) {
    env->DeleteLocalRef(obj);
}

jbyte * knn_jni::JNIUtil::GetByteArrayElements(JNIEnv *env, jbyteArray array, jboolean * isCopy) {
    jbyte * byteArray = env->GetByteArrayElements(array, nullptr);
    if (byteArray == nullptr) {
        this->HasExceptionInStack(env, "Unable able to get byte array");
        throw std::runtime_error("Unable able to get byte array");
    }

    return byteArray;
}

jfloat * knn_jni::JNIUtil::GetFloatArrayElements(JNIEnv *env, jfloatArray array, jboolean * isCopy) {
    float* floatArray = env->GetFloatArrayElements(array, nullptr);
    if (floatArray == nullptr) {
        this->HasExceptionInStack(env, "Unable to get float elements");
        throw std::runtime_error("Unable to get float elements");
    }

    return floatArray;
}

jint * knn_jni::JNIUtil::GetIntArrayElements(JNIEnv *env, jintArray array, jboolean * isCopy) {
    // Lets check for error here
    jint * intArray =  env->GetIntArrayElements(array, isCopy);
    if (intArray == nullptr) {
        this->HasExceptionInStack(env, "Unable to get int array");
        throw std::runtime_error("Unable to get int array");
    }

    return intArray;
}

jlong * knn_jni::JNIUtil::GetLongArrayElements(JNIEnv *env, jlongArray array, jboolean * isCopy) {
    // Lets check for error here
    jlong * longArray =  env->GetLongArrayElements(array, isCopy);
    if (longArray == nullptr) {
        this->HasExceptionInStack(env, "Unable to get long array");
        throw std::runtime_error("Unable to get long array");
    }

    return longArray;
}

jobject knn_jni::JNIUtil::GetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index) {
    jobject object = env->GetObjectArrayElement(array, index);
    this->HasExceptionInStack(env, "Unable to get object");
    return object;
}

jobject knn_jni::JNIUtil::NewObject(JNIEnv *env, jclass clazz, jmethodID methodId, int id, float distance) {
    jobject object = env->NewObject(clazz, methodId, id, distance);
    if (object == nullptr) {
        this->HasExceptionInStack(env, "Unable to create object");
        throw std::runtime_error("Unable to create object");
    }

    return object;
}

jobjectArray knn_jni::JNIUtil::NewObjectArray(JNIEnv *env, jsize len, jclass clazz, jobject init) {
    jobjectArray objectArray = env->NewObjectArray(len, clazz, init);
    if (objectArray == nullptr) {
        this->HasExceptionInStack(env, "Unable to allocate object array");
        throw std::runtime_error("Unable to allocate object array");
    }

    return objectArray;
}

jbyteArray knn_jni::JNIUtil::NewByteArray(JNIEnv *env, jsize len) {
    jbyteArray  byteArray = env->NewByteArray(len);
    if (byteArray == nullptr) {
        this->HasExceptionInStack(env, "Unable to allocate byte array");
        throw std::runtime_error("Unable to allocate byte array");
    }

    return byteArray;
}

void knn_jni::JNIUtil::ReleaseByteArrayElements(JNIEnv *env, jbyteArray array, jbyte *elems, int mode) {
    env->ReleaseByteArrayElements(array, elems, mode);
}

void knn_jni::JNIUtil::ReleaseFloatArrayElements(JNIEnv *env, jfloatArray array, jfloat *elems, int mode) {
    env->ReleaseFloatArrayElements(array, elems, mode);
}

void knn_jni::JNIUtil::ReleaseIntArrayElements(JNIEnv *env, jintArray array, jint *elems, jint mode) {
    env->ReleaseIntArrayElements(array, elems, mode);
}

void knn_jni::JNIUtil::ReleaseLongArrayElements(JNIEnv *env, jlongArray array, jlong *elems, jint mode) {
    env->ReleaseLongArrayElements(array, elems, mode);
}

void knn_jni::JNIUtil::SetObjectArrayElement(JNIEnv *env, jobjectArray array, jsize index, jobject val) {
    env->SetObjectArrayElement(array, index, val);
    this->HasExceptionInStack(env, "Unable to set object array element");
}

void knn_jni::JNIUtil::SetByteArrayRegion(JNIEnv *env, jbyteArray array, jsize start, jsize len, const jbyte * buf) {
    env->SetByteArrayRegion(array, start, len, buf);
    this->HasExceptionInStack(env, "Unable to set byte array region");
}

jobject knn_jni::GetJObjectFromMapOrThrow(std::unordered_map<std::string, jobject> map, std::string key) {
    if(map.find(key) == map.end()) {
        throw std::runtime_error(key + " not found");
    }
    return map[key];
}

//TODO: This potentially should use const char *
const std::string knn_jni::FAISS_NAME = "faiss";
const std::string knn_jni::NMSLIB_NAME = "nmslib";
const std::string knn_jni::ILLEGAL_ARGUMENT_PATH = "java/lang/IllegalArgumentException";

const std::string knn_jni::SPACE_TYPE = "spaceType";
const std::string knn_jni::METHOD = "method";
const std::string knn_jni::INDEX_DESCRIPTION = "index_description";
const std::string knn_jni::PARAMETERS = "parameters";
const std::string knn_jni::TRAINING_DATASET_SIZE_LIMIT = "training_dataset_size_limit";
const std::string knn_jni::INDEX_THREAD_QUANTITY = "indexThreadQty";

const std::string knn_jni::L2 = "l2";
const std::string knn_jni::L1 = "l1";
const std::string knn_jni::LINF = "linf";
const std::string knn_jni::COSINESIMIL = "cosinesimil";
const std::string knn_jni::INNER_PRODUCT = "innerproduct";
const std::string knn_jni::NEG_DOT_PRODUCT = "negdotprod";
const std::string knn_jni::HAMMING = "hamming";

const std::string knn_jni::NPROBES = "nprobes";
const std::string knn_jni::COARSE_QUANTIZER = "coarse_quantizer";
const std::string knn_jni::M = "m";
const std::string knn_jni::M_NMSLIB = "M";
const std::string knn_jni::EF_CONSTRUCTION = "ef_construction";
const std::string knn_jni::EF_CONSTRUCTION_NMSLIB = "efConstruction";
const std::string knn_jni::EF_SEARCH = "ef_search";
