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

#ifndef OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H

#include "jni_util.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni {
namespace stream {



/**
 * This class contains Java IndexInputWithBuffer reference and calls its API to copy required bytes into a read buffer.
 */
class NativeEngineIndexInputMediator {
 public:
  // Expect IndexInputWithBuffer is given as `_indexInput`.
  NativeEngineIndexInputMediator(JNIUtilInterface *_jni_interface,
                                 JNIEnv *_env,
                                 jobject _indexInput)
      : jni_interface(_jni_interface),
        env(_env),
        indexInput(_indexInput),
        bufferArray((jbyteArray) (_jni_interface->GetObjectField(_env,
                                                                 _indexInput,
                                                                 getBufferFieldId(_jni_interface, _env)))),
        copyBytesMethod(getCopyBytesMethod(_jni_interface, _env)),
        remainingBytesMethod(getRemainingBytesMethod(_jni_interface, _env)) {
  }

  void copyBytes(int64_t nbytes, uint8_t *destination) {
    auto jclazz = getIndexInputWithBufferClass(jni_interface, env);

    while (nbytes > 0) {
      // Call `copyBytes` to read bytes as many as possible.
      jvalue args;
      args.j = nbytes;
      const auto readBytes =
          jni_interface->CallNonvirtualIntMethodA(env, indexInput, jclazz, copyBytesMethod, &args);

      // === Critical Section Start ===

      // Get primitive array pointer, no copy is happening in OpenJDK.
      auto primitiveArray =
          (jbyte *) jni_interface->GetPrimitiveArrayCritical(env, bufferArray, nullptr);

      // Copy Java bytes to C++ destination address.
      std::memcpy(destination, primitiveArray, readBytes);

      // Release the acquired primitive array pointer.
      // JNI_ABORT tells JVM to directly free memory without copying back to Java byte[].
      // Since we're merely copying data, we don't need to copying back.
      jni_interface->ReleasePrimitiveArrayCritical(env, bufferArray, primitiveArray, JNI_ABORT);

      // === Critical Section End ===

      destination += readBytes;
      nbytes -= readBytes;
    }  // End while
  }

  int64_t remainingBytes() {
      return jni_interface->CallNonvirtualLongMethodA(env,
                                                      indexInput,
                                                      getIndexInputWithBufferClass(jni_interface, env),
                                                      remainingBytesMethod,
                                                      nullptr);
  }

 private:
  static jclass getIndexInputWithBufferClass(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jclass INDEX_INPUT_WITH_BUFFER_CLASS =
        jni_interface->FindClassFromJNIEnv(env, "org/opensearch/knn/index/store/IndexInputWithBuffer");
    return INDEX_INPUT_WITH_BUFFER_CLASS;
  }

  static jmethodID getCopyBytesMethod(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jmethodID COPY_METHOD_ID =
        jni_interface->GetMethodID(env, getIndexInputWithBufferClass(jni_interface, env), "copyBytes", "(J)I");
    return COPY_METHOD_ID;
  }

  static jmethodID getRemainingBytesMethod(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jmethodID COPY_METHOD_ID =
        jni_interface->GetMethodID(env, getIndexInputWithBufferClass(jni_interface, env), "remainingBytes", "()J");
    return COPY_METHOD_ID;
  }

  static jfieldID getBufferFieldId(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jfieldID BUFFER_FIELD_ID =
        jni_interface->GetFieldID(env, getIndexInputWithBufferClass(jni_interface, env), "buffer", "[B");
    return BUFFER_FIELD_ID;
  }

  JNIUtilInterface *jni_interface;
  JNIEnv *env;

  // `IndexInputWithBuffer` instance having `IndexInput` instance obtained from `Directory` for reading.
  jobject indexInput;
  jbyteArray bufferArray;
  jmethodID copyBytesMethod;
  jmethodID remainingBytesMethod;
}; // class NativeEngineIndexInputMediator



}
}

#endif //OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
