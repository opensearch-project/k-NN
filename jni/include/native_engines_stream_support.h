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
#include "parameter_utils.h"
#include "memory_util.h"

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
      : jni_interface(knn_jni::util::ParameterCheck::require_non_null(
          _jni_interface, "jni_interface")),
        env(knn_jni::util::ParameterCheck::require_non_null(_env, "env")),
        indexInput(knn_jni::util::ParameterCheck::require_non_null(_indexInput, "indexInput")),
        bufferArray((jbyteArray) (_jni_interface->GetObjectField(_env,
                                                                 _indexInput,
                                                                 getBufferFieldId(_jni_interface, _env)))),
        copyBytesMethod(getCopyBytesMethod(_jni_interface, _env)),
        remainingBytesMethod(getRemainingBytesMethod(_jni_interface, _env)) {
  }

  void copyBytes(int64_t nbytes, uint8_t * RESTRICT destination) {
    auto jclazz = getIndexInputWithBufferClass(jni_interface, env);

    while (nbytes > 0) {
      // Call `copyBytes` to read bytes as many as possible.
      jvalue args;
      args.j = nbytes;
      const auto readBytes =
          jni_interface->CallNonvirtualIntMethodA(env, indexInput, jclazz, copyBytesMethod, &args);
      jni_interface->HasExceptionInStack(env, "Reading bytes via IndexInput has failed.");

      // === Critical Section Start ===

      // Get primitive array pointer, no copy is happening in OpenJDK.
      jbyte * RESTRICT primitiveArray =
          (jbyte *) jni_interface->GetPrimitiveArrayCritical(env, bufferArray, nullptr);

      // Copy Java bytes to C++ destination address.
      std::memcpy(destination, primitiveArray, readBytes);

      // Release the acquired primitive array pointer.
      // JNI_ABORT tells JVM to directly free memory without copying back to Java byte[].
      // Since we're merely copying data, we don't need to copying back.
      // Note than when we received an internal primitive array pointer, then the mode will be ignored.
      jni_interface->ReleasePrimitiveArrayCritical(env, bufferArray, primitiveArray, JNI_ABORT);

      // === Critical Section End ===

      destination += readBytes;
      nbytes -= readBytes;
    }  // End while
  }

  int64_t remainingBytes() {
    auto bytes = jni_interface->CallNonvirtualLongMethodA(env,
                                                          indexInput,
                                                          getIndexInputWithBufferClass(jni_interface, env),
                                                          remainingBytesMethod,
                                                          nullptr);
    jni_interface->HasExceptionInStack(env, "Checking remaining bytes has failed.");
    return bytes;
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



/**
 * This class delegates the provided index output to do IO processing.
 * In most cases, it is expected that IndexOutputWithBuffer was passed down to this,
 * which eventually have Lucene's IndexOutput to write bytes.
 */
class NativeEngineIndexOutputMediator {
 public:
  NativeEngineIndexOutputMediator(JNIUtilInterface *_jni_interface,
                                  JNIEnv *_env,
                                  jobject _indexOutput)
      : jni_interface(knn_jni::util::ParameterCheck::require_non_null(_jni_interface, "jni_interface")),
        env(knn_jni::util::ParameterCheck::require_non_null(_env, "env")),
        indexOutput(knn_jni::util::ParameterCheck::require_non_null(_indexOutput, "indexOutput")),
        bufferArray((jbyteArray) (_jni_interface->GetObjectField(_env,
                                                                 _indexOutput,
                                                                 getBufferFieldId(_jni_interface, _env)))),
        writeBytesMethod(getWriteBytesMethod(_jni_interface, _env)),
        bufferLength(jni_interface->GetJavaBytesArrayLength(env, bufferArray)),
        nextWriteIndex() {
  }

  void writeBytes(const uint8_t * RESTRICT source, size_t nbytes) {
    auto left = nbytes;
    while (left > 0) {
      const auto writeBytes = std::min(bufferLength - nextWriteIndex, left);

      // === Critical Section Start ===

      // Get primitive array pointer, no copy is happening in OpenJDK.
      jbyte * RESTRICT primitiveArray =
          (jbyte *) jni_interface->GetPrimitiveArrayCritical(env, bufferArray, nullptr);

      // Copy the given bytes to Java byte[] address.
      std::memcpy(primitiveArray + nextWriteIndex, source, writeBytes);

      // Release the acquired primitive array pointer.
      // 0 tells JVM to copy back the content, and to free the pointer. It will be ignored if we acquired an internal
      // primitive array pointer instead of a copied version.
      // From JNI docs:
      // Mode 0 : copy back the content and free the elems buffer
      // The mode argument provides information on how the array buffer should be released. mode has no effect if elems
      // is not a copy of the elements in array.
      jni_interface->ReleasePrimitiveArrayCritical(env, bufferArray, primitiveArray, 0);

      // === Critical Section End ===

      nextWriteIndex += writeBytes;
      if (nextWriteIndex >= bufferLength) {
        callWriteBytesInIndexOutput();
      }

      source += writeBytes;
      left -= writeBytes;
    }  // End while
  }

  void flush() {
    if (nextWriteIndex > 0) {
      callWriteBytesInIndexOutput();
    }
  }

 private:
  static jclass getIndexOutputWithBufferClass(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jclass INDEX_OUTPUT_WITH_BUFFER_CLASS =
        jni_interface->FindClassFromJNIEnv(env, "org/opensearch/knn/index/store/IndexOutputWithBuffer");
    return INDEX_OUTPUT_WITH_BUFFER_CLASS;
  }

  static jmethodID getWriteBytesMethod(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jmethodID WRITE_METHOD_ID =
        jni_interface->GetMethodID(env, getIndexOutputWithBufferClass(jni_interface, env), "writeBytes", "(I)V");
    return WRITE_METHOD_ID;
  }

  static jfieldID getBufferFieldId(JNIUtilInterface *jni_interface, JNIEnv *env) {
    static jfieldID BUFFER_FIELD_ID =
        jni_interface->GetFieldID(env, getIndexOutputWithBufferClass(jni_interface, env), "buffer", "[B");
    return BUFFER_FIELD_ID;
  }

  void callWriteBytesInIndexOutput() {
    auto jclazz = getIndexOutputWithBufferClass(jni_interface, env);
    // Initializing the first integer parameter of `writeBytes`.
    // `i` represents an integer parameter.
    jvalue args {.i = nextWriteIndex};
    jni_interface->CallNonvirtualVoidMethodA(env, indexOutput, jclazz, writeBytesMethod, &args);
    jni_interface->HasExceptionInStack(env, "Writing bytes via IndexOutput has failed.");
    nextWriteIndex = 0;
  }

  JNIUtilInterface *jni_interface;
  JNIEnv *env;

  // `IndexOutputWithBuffer` instance having `IndexOutput` instance obtained from `Directory` for reading.
  jobject indexOutput;
  jbyteArray bufferArray;
  jmethodID writeBytesMethod;
  size_t bufferLength;
  int32_t nextWriteIndex;
};  // NativeEngineIndexOutputMediator



}
}

#endif //OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
