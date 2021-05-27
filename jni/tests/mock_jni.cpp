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

#include <jni.h>
#include "mock_jni.h"



JNINativeInterface_ * mock_jni::GenerateMockJNINativeInterface() {
    auto * jniNativeInterface = new JNINativeInterface_{
            nullptr, nullptr, nullptr, nullptr
    };

    jniNativeInterface->FindClass = reinterpret_cast<jclass (*)(JNIEnv *, const char *)>(
            *[](JNIEnv_ jniEnv, const char *classname) {
                return (jclass) 1;
            }
    );

    jniNativeInterface->GetMethodID = reinterpret_cast<jmethodID (*)(JNIEnv *, jclass, const char *, const char *)>(
            *[](JNIEnv_ jniEnv, jclass clazz, const char *name, const char *sig) {
                return (jmethodID) 1;
            }
    );

    jniNativeInterface->ExceptionCheck = reinterpret_cast<jboolean (*)(JNIEnv *)>(
            *[](JNIEnv_ jniEnv) {
                return false;
            }
    );

    jniNativeInterface->ReleaseStringUTFChars = reinterpret_cast<void (*)(JNIEnv *, jstring, const char *)>(
            *[](JNIEnv_ jniEnv, jstring str, const char *chars) {}
    );

    jniNativeInterface->ReleaseIntArrayElements = reinterpret_cast<void (*)(JNIEnv *, jintArray , jint *, jint)>(
            *[](JNIEnv_ jniEnv, jintArray array, jint * elems, jint mode) {}
    );

    jniNativeInterface->ReleaseFloatArrayElements = reinterpret_cast<void (*)(JNIEnv *, jfloatArray , jfloat *, jint)>(
            *[](JNIEnv_ jniEnv, jfloatArray array, jfloat * elems, jint mode) {}
    );

    jniNativeInterface->GetStringUTFChars = reinterpret_cast<const char *(*)(JNIEnv *, jstring, jboolean *)>(
            *[](JNIEnv_ jniEnv, jstring str, jboolean *isCopy) {
                return (const char *) str;
            }
    );

    jniNativeInterface->GetArrayLength = reinterpret_cast<jsize (*)(JNIEnv *, jarray)>(
            *[](JNIEnv_ jniEnv, jintArray array) {
                auto *cppVector = reinterpret_cast<std::vector<int> *>(array);
                return cppVector->size();
            }
    );

    jniNativeInterface->GetFloatArrayElements = reinterpret_cast<jfloat *(*)(JNIEnv *, jfloatArray, jboolean *)>(
            *[](JNIEnv_ jniEnv, jfloatArray array, jboolean *isCopy) {
                auto *cppVector = reinterpret_cast<std::vector<float> *>(array);
                return (jfloat *) cppVector->data();
            }
    );

    jniNativeInterface->GetIntArrayElements = reinterpret_cast<jint *(*)(JNIEnv *, jintArray, jboolean *)>(
            *[](JNIEnv_ jniEnv, jintArray array, jboolean *isCopy) {
                auto *cppVector = reinterpret_cast<std::vector<int> *>(array);
                return (jint *) cppVector->data();
            }
    );

    jniNativeInterface->DeleteLocalRef = reinterpret_cast<void (*)(JNIEnv *, jobject)>(
            *[](JNIEnv_ jniEnv, jobject obj) {}
    );

    return jniNativeInterface;
}
