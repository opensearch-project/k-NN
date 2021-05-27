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

#include <iostream>
#include <utility>
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

    return jniNativeInterface;
}

jsize mock_jni::MockGetArrayLength(jarray array) {
    std::cout << "Value of javaIntArray after: " << array << std::endl;

    auto * intArray = reinterpret_cast<std::vector<int> *>(array);

    return (intArray)->size();
}


jint mock_jni::MockThrow(jthrowable jthrowable1) {
    std::cout << "Jthrow address after: " << jthrowable1 << std::endl;
    return 0;
}

jboolean mock_jni::MockExceptionCheck() {
    return false;
}