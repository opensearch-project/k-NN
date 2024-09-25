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

#include "faiss/impl/io.h"
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
        copyBytesMethod(getCopyBytesMethod(_jni_interface, _env)) {
  }

  void copyBytes(int32_t nbytes, uint8_t *destination) {
    while (nbytes > 0) {
      // Call `copyBytes` to read bytes as many as possible.
      const auto readBytes =
          jni_interface->CallIntMethodInt(env, indexInput, copyBytesMethod, nbytes);

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
}; // class NativeEngineIndexInputMediator



/**
 * A glue component inheriting IOReader to be passed down to Faiss library.
 * This will then indirectly call the mediator component and eventually read required bytes from Lucene's IndexInput.
 */
class FaissOpenSearchIOReader final : public faiss::IOReader {
 public:
  explicit FaissOpenSearchIOReader(NativeEngineIndexInputMediator *_mediator)
      : faiss::IOReader(),
        mediator(_mediator) {
    name = "FaissOpenSearchIOReader";
  }

  size_t operator()(void *ptr, size_t size, size_t nitems) final {
    const auto readBytes = size * nitems;
    if (readBytes > 0) {
      // Mediator calls IndexInput, then copy read bytes to `ptr`.
      mediator->copyBytes(readBytes, (uint8_t *) ptr);
    }
    return nitems;
  }

  int filedescriptor() final {
    throw std::runtime_error("filedescriptor() is not supported in FaissOpenSearchIOReader.");
  }

 private:
  NativeEngineIndexInputMediator *mediator;
};  // class FaissOpenSearchIOReader



}
}

#endif //OPENSEARCH_KNN_JNI_STREAM_SUPPORT_H
